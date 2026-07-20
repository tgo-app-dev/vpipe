#include "generative-models/qwen-image/metal-qwen25-vision.h"

#include "generative-models/llama3/metal-llama-weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

inline std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t u; std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// Qwen2.5-VL vision window partition for a single image grid (matches HF
// Qwen2_5_VisionTransformer.get_window_index). `gh`x`gw` are the PATCH grid;
// `merge` the spatial-merge (2); `wm` the window size in MERGED tokens (4).
// Fills `window_index` (mseq entries, the merged-unit reorder) and `cu` (patch-
// unit window boundaries in the reordered sequence, len = nwindows+1). Window
// tiles that overhang the grid edge are smaller (their -100 pads dropped).
void
compute_windows_(int gh, int gw, int merge, int wm,
                 std::vector<int>& window_index, std::vector<int>& cu)
{
  const int lh = gh / merge, lw = gw / merge;   // llm (merged) grid
  const int MU = merge * merge;
  const int pad_h = (wm - lh % wm) % wm;
  const int pad_w = (wm - lw % wm) % wm;
  const int nwh = (lh + pad_h) / wm, nww = (lw + pad_w) / wm;
  window_index.clear();
  cu.clear();
  cu.push_back(0);
  for (int wh = 0; wh < nwh; ++wh) {
    for (int ww = 0; ww < nww; ++ww) {
      int cnt = 0;
      for (int ih = 0; ih < wm; ++ih) {
        for (int iw = 0; iw < wm; ++iw) {
          const int r = wh * wm + ih, c = ww * wm + iw;
          if (r < lh && c < lw) {
            window_index.push_back(r * lw + c);
            ++cnt;
          }
        }
      }
      cu.push_back(cu.back() + cnt * MU);
    }
  }
}

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

MetalQwen25Vision::~MetalQwen25Vision() = default;

SharedBuffer
MetalQwen25Vision::to_elt_(const MetalLlamaWeights& wts, const std::string& name)
{
  return vpipe::genai::to_elt_(wts, _mc, name);
}

bool
MetalQwen25Vision::load_linear_(const MetalLlamaWeights& wts,
                                const std::string& pre, SharedBuffer& w,
                                SharedBuffer& b)
{
  w = to_elt_(wts, pre + ".weight");
  b = to_elt_(wts, pre + ".bias");
  return !w.empty() && !b.empty();
}

std::unique_ptr<MetalQwen25Vision>
MetalQwen25Vision::load(const std::string& model_dir, MetalCompute* mc,
                        const Config& cfg)
{
  if (mc == nullptr) { return nullptr; }
  auto wtsopt = MetalLlamaWeights::open_model(model_dir);
  if (!wtsopt.has_value()) { return nullptr; }
  const MetalLlamaWeights& wts = *wtsopt;
  auto m = std::unique_ptr<MetalQwen25Vision>(new MetalQwen25Vision());
  m->_mc = mc; m->_cfg = cfg;

  m->_lib_gemm = mc->load_library("dense_gemm_bf16");
  m->_lib_elt  = mc->load_library("llm_elementwise_bf16");
  m->_lib_rms  = mc->load_library("rms_norm_bf16");
  m->_lib_sdpa = mc->load_library("sdpa_bf16");
  m->_lib_rope = mc->load_library("rope_bf16");
  m->_fn_gemm      = m->_lib_gemm.function("dense_gemm_t_f16");
  m->_fn_gemm_bias = m->_lib_gemm.function("dense_gemm_bias_f16");
  m->_fn_rms       = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_silu      = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_swiglu    = m->_lib_elt.function("swiglu_f16");
  m->_fn_mul       = m->_lib_elt.function("head_slice_f16");
  m->_fn_gelu      = m->_lib_elt.function("gelu_tanh_ff_f16");
  m->_fn_residual  = m->_lib_elt.function("residual_add_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_sdpa      = m->_lib_sdpa.function("sdpa_full_f16");
  m->_fn_varwindow = m->_lib_sdpa.function("sdpa_varwindow_f16");
  m->_fn_rope      = m->_lib_rope.function("rope_half_table_ftab_f16");
  if (!m->_fn_gemm.valid() || !m->_fn_gemm_bias.valid() || !m->_fn_rms.valid() ||
      !m->_fn_silu.valid() || !m->_fn_swiglu.valid() || !m->_fn_mul.valid() ||
      !m->_fn_gelu.valid() || !m->_fn_residual.valid() ||
      !m->_fn_transpose.valid() || !m->_fn_sdpa.valid() ||
      !m->_fn_varwindow.valid() || !m->_fn_rope.valid()) {
    return nullptr;
  }

  // patch_embed.proj is a Conv3d [hidden,3,2,14,14] == a per-patch Linear
  // [hidden, patch_in] (stride == kernel). No bias.
  m->_patch_w = m->to_elt_(wts, "visual.patch_embed.proj.weight");
  bool ok = !m->_patch_w.empty();
  m->_merge_ln = m->to_elt_(wts, "visual.merger.ln_q.weight");
  ok = ok && !m->_merge_ln.empty();
  ok = ok && m->load_linear_(wts, "visual.merger.mlp.0", m->_merge0_w, m->_merge0_b);
  ok = ok && m->load_linear_(wts, "visual.merger.mlp.2", m->_merge2_w, m->_merge2_b);
  if (!ok) { return nullptr; }

  m->_blocks.resize((std::size_t)cfg.depth);
  for (int L = 0; L < cfg.depth; ++L) {
    Block& b = m->_blocks[(std::size_t)L];
    const std::string p = "visual.blocks." + std::to_string(L) + ".";
    b.n1 = m->to_elt_(wts, p + "norm1.weight");
    b.n2 = m->to_elt_(wts, p + "norm2.weight");
    bool bok = !b.n1.empty() && !b.n2.empty();
    bok = bok && m->load_linear_(wts, p + "attn.qkv", b.qkv_w, b.qkv_b);
    bok = bok && m->load_linear_(wts, p + "attn.proj", b.proj_w, b.proj_b);
    bok = bok && m->load_linear_(wts, p + "mlp.gate_proj", b.gw, b.gb);
    bok = bok && m->load_linear_(wts, p + "mlp.up_proj", b.uw, b.ub);
    bok = bok && m->load_linear_(wts, p + "mlp.down_proj", b.dw, b.db);
    if (!bok) { return nullptr; }
  }
  return m;
}

void
MetalQwen25Vision::build_rope_(int seq, const std::vector<int>& pos,
                               SharedBuffer& cos_out, SharedBuffer& sin_out)
{
  const int D = _cfg.head_dim;      // 80
  const int half = D / 2;           // 40 (h freqs [0:20], w freqs [20:40])
  const int nf = half / 2;          // 20 freqs per axis
  cos_out = _mc->make_shared_buffer((std::size_t)seq * half * 4);   // f32
  sin_out = _mc->make_shared_buffer((std::size_t)seq * half * 4);
  auto* cb = static_cast<float*>(cos_out.contents());
  auto* sb = static_cast<float*>(sin_out.contents());
  for (int t = 0; t < seq; ++t) {
    const double hp = (double)pos[(std::size_t)t * 2];
    const double wp = (double)pos[(std::size_t)t * 2 + 1];
    for (int j = 0; j < half; ++j) {
      const int axis_j = (j < nf) ? j : (j - nf);
      const double p = (j < nf) ? hp : wp;
      // inv_freq for Qwen2_5_VisionRotaryEmbedding(dim = head_dim/2 = 40):
      // 1/theta^(2*axis_j / 40).
      const double invf =
          std::pow((double)_cfg.rope_theta, -2.0 * (double)axis_j / (double)half);
      const double ang = p * invf;
      const std::size_t o = (std::size_t)t * half + j;
      cb[o] = (float)std::cos(ang);
      sb[o] = (float)std::sin(ang);
    }
  }
}

SharedBuffer
MetalQwen25Vision::encode_rgb(const std::uint8_t* rgb, int H, int W,
                              int cond_area, int& grid_h, int& grid_w)
{
  if (rgb == nullptr || H <= 0 || W <= 0) { return {}; }
  const int P = 14, T = 2, M = _cfg.merge;   // patch, temporal, merge
  const int factor = P * M;                  // 28
  // QwenImageEditPlus condition dims: resize to `cond_area`, then smart_resize
  // to a multiple of `factor` (aspect from the ORIGINAL image).
  const double ratio = (double)W / (double)H;
  auto round32 = [](double v) { return (int)std::round(v / 32.0) * 32; };
  // calc_dims: round the UNROUNDED w and h independently (h from unrounded w).
  const double w_un = std::sqrt((double)cond_area * ratio);
  const double h_un = w_un / ratio;                 // = sqrt(cond_area / ratio)
  const int cw = std::max(32, round32(w_un));
  const int ch = std::max(32, round32(h_un));
  // smart_resize(ch, cw) -> (th, tw); condition dims sit between min/max so this
  // is just rounding to `factor`.
  int th = std::max(factor, (int)std::round((double)ch / factor) * factor);
  int tw = std::max(factor, (int)std::round((double)cw / factor) * factor);
  const int gh = th / P, gw = tw / P;        // patch grid
  const int seq = gh * gw;
  grid_h = gh; grid_w = gw;

  // CLIP (OpenAI) mean/std -- the Qwen2.5-VL image_mean/std.
  static const float MEAN[3] = {0.48145466f, 0.4578275f, 0.40821073f};
  static const float STD[3]  = {0.26862954f, 0.26130258f, 0.27577711f};
  // Bilinear resize [3,H,W] U8 -> normalized [3,th,tw] f32 (channel-first).
  std::vector<float> norm((std::size_t)3 * th * tw);
  { const double sy = (double)H / th, sx = (double)W / tw;
    const int pin = H * W, pout = th * tw;
    for (int c = 0; c < 3; ++c) {
      const std::uint8_t* src = rgb + (std::size_t)c * pin;
      float* dst = norm.data() + (std::size_t)c * pout;
      const float inv = 1.0f / (255.0f * STD[c]), bias = -MEAN[c] / STD[c];
      for (int yy = 0; yy < th; ++yy) {
        double y = std::min(std::max((yy + 0.5) * sy - 0.5, 0.0), (double)(H - 1));
        int y0 = (int)std::floor(y), y1 = std::min(y0 + 1, H - 1);
        float dy = (float)(y - y0);
        for (int xx = 0; xx < tw; ++xx) {
          double x = std::min(std::max((xx + 0.5) * sx - 0.5, 0.0), (double)(W - 1));
          int x0 = (int)std::floor(x), x1 = std::min(x0 + 1, W - 1);
          float dx = (float)(x - x0);
          float v0 = src[y0 * W + x0] * (1 - dx) + src[y0 * W + x1] * dx;
          float v1 = src[y1 * W + x0] * (1 - dx) + src[y1 * W + x1] * dx;
          dst[yy * tw + xx] = (v0 * (1 - dy) + v1 * dy) * inv + bias;
        }
      }
    }
  }
  // Patchify to merge-block order [gh//M, gw//M, M, M] with per-patch features
  // [c, t, py, px] (temporal frame repeated) -> [seq, patch_in] bf16. Positions
  // (h,w) in the same order.
  SharedBuffer pixels = _mc->make_shared_buffer((std::size_t)seq * _cfg.patch_in * 2);
  auto* pd = static_cast<std::uint16_t*>(pixels.contents());
  std::vector<int> pos((std::size_t)seq * 2);
  const int lh = gh / M, lw = gw / M;
  int t_idx = 0;
  for (int mr = 0; mr < lh; ++mr) {
    for (int mc = 0; mc < lw; ++mc) {
      for (int ih = 0; ih < M; ++ih) {
        for (int iw = 0; iw < M; ++iw) {
          const int py0 = (mr * M + ih) * P, px0 = (mc * M + iw) * P;
          std::uint16_t* row = pd + (std::size_t)t_idx * _cfg.patch_in;
          int f = 0;
          for (int c = 0; c < 3; ++c) {
            const float* ch_p = norm.data() + (std::size_t)c * th * tw;
            for (int tt = 0; tt < T; ++tt) {          // temporal frame repeat
              for (int yy = 0; yy < P; ++yy) {
                for (int xx = 0; xx < P; ++xx) {
                  row[f++] = f32_to_bf16_(ch_p[(py0 + yy) * tw + (px0 + xx)]);
                }
              }
            }
          }
          pos[(std::size_t)t_idx * 2] = mr * M + ih;
          pos[(std::size_t)t_idx * 2 + 1] = mc * M + iw;
          ++t_idx;
        }
      }
    }
  }
  return encode(pixels, seq, pos);
}

SharedBuffer
MetalQwen25Vision::encode(const SharedBuffer& pixels, int seq,
                          const std::vector<int>& pos)
{
  const int H = _cfg.hidden, Hd = _cfg.head_dim, NH = _cfg.n_heads;
  const int FF = _cfg.ffn, OUT = _cfg.out_hidden, M = _cfg.merge;
  const int MU = M * M, mseq = seq / MU, mh = H * MU;   // merged
  const float eps = _cfg.norm_eps;

  auto buf = [&](std::size_t n) { return _mc->make_shared_buffer(n * 2); };

  // ---- Window partition. Recover the patch grid from `pos` (per-patch (h,w)),
  // build the merged-unit window reorder, and reorder the pixels + positions so
  // each attention window is contiguous. 28/32 blocks then attend within-window
  // (varwindow sdpa over [win_start,win_end)); blocks {7,15,23,31} attend fully.
  // A single-window grid yields identity perm + one full window == the old path.
  int gh = 0, gw = 0;
  for (int i = 0; i < seq; ++i) {
    if (pos[(std::size_t)2 * i] + 1 > gh)     { gh = pos[(std::size_t)2 * i] + 1; }
    if (pos[(std::size_t)2 * i + 1] + 1 > gw) { gw = pos[(std::size_t)2 * i + 1] + 1; }
  }
  std::vector<int> window_index, cu;
  compute_windows_(gh, gw, M, _cfg.window_merge, window_index, cu);
  const bool reorder = ((int)window_index.size() == mseq) && (cu.size() > 2);
  std::vector<int> perm(seq);          // perm[new] = old patch index
  std::vector<int> win_s(seq), win_e(seq);
  if (reorder) {
    for (int i = 0; i < mseq; ++i) {
      for (int j = 0; j < MU; ++j) { perm[(std::size_t)i * MU + j] =
                                         window_index[i] * MU + j; }
    }
    for (std::size_t w = 0; w + 1 < cu.size(); ++w) {
      for (int p = cu[w]; p < cu[w + 1]; ++p) { win_s[p] = cu[w];
                                                win_e[p] = cu[w + 1]; }
    }
  } else {
    for (int i = 0; i < seq; ++i) { perm[i] = i; win_s[i] = 0; win_e[i] = seq; }
  }

  // Reorder pixels rows + positions by perm (host; pixels is bf16 [seq,patch_in]).
  SharedBuffer pix_r = pixels.subview(0, pixels.byte_size());  // alias unless reordered
  std::vector<int> pos_r = pos;
  if (reorder) {
    pix_r = buf((std::size_t)seq * _cfg.patch_in);
    const auto* ps = static_cast<const std::uint16_t*>(pixels.contents());
    auto* pd = static_cast<std::uint16_t*>(pix_r.contents());
    for (int i = 0; i < seq; ++i) {
      std::memcpy(pd + (std::size_t)i * _cfg.patch_in,
                  ps + (std::size_t)perm[i] * _cfg.patch_in,
                  (std::size_t)_cfg.patch_in * 2);
    }
    for (int i = 0; i < seq; ++i) {
      pos_r[(std::size_t)2 * i] = pos[(std::size_t)2 * perm[i]];
      pos_r[(std::size_t)2 * i + 1] = pos[(std::size_t)2 * perm[i] + 1];
    }
  }
  // win_start/win_end int buffers for the varwindow sdpa (4 bytes each).
  SharedBuffer win_sb = _mc->make_shared_buffer((std::size_t)seq * 4);
  SharedBuffer win_eb = _mc->make_shared_buffer((std::size_t)seq * 4);
  std::memcpy(win_sb.contents(), win_s.data(), (std::size_t)seq * 4);
  std::memcpy(win_eb.contents(), win_e.data(), (std::size_t)seq * 4);

  SharedBuffer rcos, rsin;
  build_rope_(seq, pos_r, rcos, rsin);

  SharedBuffer x = buf((std::size_t)seq * H);
  SharedBuffer nrm = buf((std::size_t)seq * H);
  SharedBuffer qkv = buf((std::size_t)seq * 3 * H);
  SharedBuffer q = buf((std::size_t)seq * H), k = buf((std::size_t)seq * H),
               v = buf((std::size_t)seq * H);
  SharedBuffer qt = buf((std::size_t)seq * H), kt = buf((std::size_t)seq * H),
               vt = buf((std::size_t)seq * H), at = buf((std::size_t)seq * H);
  SharedBuffer att = buf((std::size_t)seq * H), o = buf((std::size_t)seq * H);
  SharedBuffer g = buf((std::size_t)seq * FF), u = buf((std::size_t)seq * FF);
  SharedBuffer mn = buf((std::size_t)seq * H);
  SharedBuffer m0 = buf((std::size_t)mseq * mh);
  SharedBuffer tok = buf((std::size_t)mseq * OUT);

  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto gemm = [&](const SharedBuffer& xb, std::size_t xe, const SharedBuffer& w,
                    const SharedBuffer& y, std::size_t ye, int Mm, int N, int K) {
      enc.set_function(_fn_gemm);
      enc.set_buffer(0, xb, xe * 2); enc.set_buffer(1, w); enc.set_buffer(2, w);
      enc.set_buffer(3, y, ye * 2);
      enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, Mm);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                    (unsigned)(((Mm + 31) / 32) * 2), 2}, {32, 2, 2});
    };
    auto gemm_bias = [&](const SharedBuffer& xb, std::size_t xe,
                         const SharedBuffer& w, const SharedBuffer& bs,
                         const SharedBuffer& y, std::size_t ye, int Mm, int N,
                         int K) {
      enc.set_function(_fn_gemm_bias);
      enc.set_buffer(0, xb, xe * 2); enc.set_buffer(1, w); enc.set_buffer(2, bs);
      enc.set_buffer(3, y, ye * 2);
      enc.set_constant(4, Mm); enc.set_constant(5, N); enc.set_constant(6, K);
      enc.set_constant(7, 1);
      enc.dispatch({(unsigned)(((N + 15) / 16) * 16),
                    (unsigned)(((Mm + 15) / 16) * 16), 1}, {16, 16, 1});
    };
    auto rms = [&](const SharedBuffer& xb, std::size_t xe, const SharedBuffer& w,
                   const SharedBuffer& y, std::size_t ye, int R, int Dd) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, xb, xe * 2); enc.set_buffer(1, w);
      enc.set_buffer(2, y, ye * 2);
      enc.set_constant(3, Dd); enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    auto hslice = [&](const SharedBuffer& in, const SharedBuffer& out, int R,
                      int stride, int width, int off) {
      enc.set_function(_fn_mul);
      enc.set_buffer(0, in); enc.set_buffer(1, out);
      enc.set_constant(2, R); enc.set_constant(3, stride); enc.set_constant(4, width);
      enc.set_constant(5, off); enc.set_constant(6, 0); enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(R * width), 1, 1}, {256, 1, 1});
    };
    auto transpose = [&](const SharedBuffer& in, const SharedBuffer& out, int A,
                         int Bd) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, in); enc.set_buffer(1, out);
      enc.set_constant(2, A); enc.set_constant(3, Bd); enc.set_constant(4, Hd);
      enc.dispatch({(unsigned)Hd, (unsigned)Bd, (unsigned)A}, {(unsigned)Hd, 1, 1});
    };
    auto rope = [&](const SharedBuffer& xb) {
      enc.set_function(_fn_rope);
      enc.set_buffer(0, xb); enc.set_buffer(1, rcos); enc.set_buffer(2, rsin);
      enc.set_constant(3, NH); enc.set_constant(4, seq); enc.set_constant(5, Hd);
      enc.dispatch({(unsigned)(Hd / 2), (unsigned)seq, (unsigned)NH},
                   {(unsigned)(Hd / 2), 1, 1});
    };
    auto sdpa = [&](const SharedBuffer& qb, const SharedBuffer& kb,
                    const SharedBuffer& vb, const SharedBuffer& out) {
      const float scale = 1.0f / std::sqrt((float)Hd);
      enc.set_function(_fn_sdpa);
      enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
      enc.set_buffer(3, out);
      enc.set_constant(4, scale); enc.set_constant(5, seq); enc.set_constant(6, Hd);
      enc.set_constant(7, NH); enc.set_constant(8, NH); enc.set_constant(9, seq);
      enc.set_constant(10, seq);
      enc.dispatch({32, (unsigned)NH, (unsigned)seq}, {32, 1, 1});
    };
    // Windowed attention: each query attends only within [win_start,win_end).
    auto sdpa_window = [&](const SharedBuffer& qb, const SharedBuffer& kb,
                           const SharedBuffer& vb, const SharedBuffer& out) {
      const float scale = 1.0f / std::sqrt((float)Hd);
      enc.set_function(_fn_varwindow);
      enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
      enc.set_buffer(3, out);
      enc.set_constant(4, scale); enc.set_constant(5, seq); enc.set_constant(6, Hd);
      enc.set_constant(7, NH); enc.set_constant(8, NH); enc.set_constant(9, seq);
      enc.set_constant(10, seq);
      enc.set_buffer(11, win_sb); enc.set_buffer(12, win_eb);
      enc.dispatch({32, (unsigned)NH, (unsigned)seq}, {32, 1, 1});
    };
    auto elt = [&](const metal_compute::ComputeFunction& fn, const SharedBuffer& a,
                   const SharedBuffer& b2, const SharedBuffer& out, int nn) {
      enc.set_function(fn);
      enc.set_buffer(0, a); enc.set_buffer(1, b2); enc.set_buffer(2, out);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto gelu = [&](const SharedBuffer& xb, const SharedBuffer& y, int nn) {
      enc.set_function(_fn_gelu);
      enc.set_buffer(0, xb); enc.set_buffer(1, y); enc.set_constant(2, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };

    // patch embed (Linear, no bias): [seq,patch_in] -> [seq,H] (reordered).
    gemm(pix_r, 0, _patch_w, x, 0, seq, H, _cfg.patch_in);

    const char* dbg = std::getenv("VPIPE_QIE_VIS_STOP");
    const int nrun = dbg ? (std::atoi(dbg) + 1) : _cfg.depth;   // -1->0 blocks
    for (int L = 0; L < nrun && L < _cfg.depth; ++L) {
      const Block& b = _blocks[(std::size_t)L];
      // attn
      rms(x, 0, b.n1, nrm, 0, seq, H);
      gemm_bias(nrm, 0, b.qkv_w, b.qkv_b, qkv, 0, seq, 3 * H, H);
      hslice(qkv, q, seq, 3 * H, H, 0);
      hslice(qkv, k, seq, 3 * H, H, H);
      hslice(qkv, v, seq, 3 * H, H, 2 * H);
      transpose(q, qt, seq, NH);   // [seq,NH,Hd] -> [NH,seq,Hd]
      transpose(k, kt, seq, NH);
      transpose(v, vt, seq, NH);
      rope(qt); rope(kt);
      // Full attention on the fullatt blocks {7,15,23,31}; windowed elsewhere.
      const bool full = !reorder || L == 7 || L == 15 || L == 23 || L == 31;
      if (full) { sdpa(qt, kt, vt, at); }
      else      { sdpa_window(qt, kt, vt, at); }
      transpose(at, att, NH, seq);   // [NH,seq,Hd] -> [seq,NH,Hd]
      gemm_bias(att, 0, b.proj_w, b.proj_b, o, 0, seq, H, H);
      elt(_fn_residual, x, o, x, seq * H);
      // mlp (SwiGLU)
      rms(x, 0, b.n2, nrm, 0, seq, H);
      gemm_bias(nrm, 0, b.gw, b.gb, g, 0, seq, FF, H);
      gemm_bias(nrm, 0, b.uw, b.ub, u, 0, seq, FF, H);
      elt(_fn_swiglu, g, u, g, seq * FF);
      gemm_bias(g, 0, b.dw, b.db, o, 0, seq, H, FF);
      elt(_fn_residual, x, o, x, seq * H);
    }

    if (dbg) {   // debug: return the [seq,H] hidden (pre-merger) after nrun blocks
      enc.end(); stream.commit().wait();
      SharedBuffer h = buf((std::size_t)seq * H);
      std::memcpy(h.contents(), x.contents(), (std::size_t)seq * H * 2);
      return h;
    }
    // merger: RMSNorm(H) -> reshape [mseq, mh] -> Linear -> GELU -> Linear.
    rms(x, 0, _merge_ln, mn, 0, seq, H);
    gemm_bias(mn, 0, _merge0_w, _merge0_b, m0, 0, mseq, mh, mh);
    gelu(m0, m0, mseq * mh);
    gemm_bias(m0, 0, _merge2_w, _merge2_b, tok, 0, mseq, OUT, mh);
  }
  stream.commit().wait();
  if (!reorder) { return tok; }
  // Reverse the window reorder: the merger ran in reordered space, so scatter
  // merged token i back to its original position window_index[i].
  SharedBuffer out = buf((std::size_t)mseq * OUT);
  const auto* ts = static_cast<const std::uint16_t*>(tok.contents());
  auto* od = static_cast<std::uint16_t*>(out.contents());
  for (int i = 0; i < mseq; ++i) {
    std::memcpy(od + (std::size_t)window_index[i] * OUT,
                ts + (std::size_t)i * OUT, (std::size_t)OUT * 2);
  }
  return out;
}

}  // namespace genai
}  // namespace vpipe
