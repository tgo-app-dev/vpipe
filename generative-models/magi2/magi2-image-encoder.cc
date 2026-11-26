#include "generative-models/magi2/magi2-image-encoder.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/magi2/magi2-latent.h"
#include "generative-models/minimax-h3/minimax-h3-references.h"
#include "generative-models/weight-set.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vpipe {
namespace genai {

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// An activation: channel-last [H, W, C] f32.
struct Act {
  int H = 0, W = 0, C = 0;
  std::vector<float> v;
  float* px(int y, int x) { return v.data() + ((std::size_t)y * W + x) * C; }
  const float* px(int y, int x) const
  {
    return v.data() + ((std::size_t)y * W + x) * C;
  }
};

// A conv as a GEMM: w [cout, taps * cin] (tap-major), bias [cout].
struct Conv {
  int cin = 0, cout = 0, k = 1;
  std::vector<float> w, b;
};

struct Res {
  std::vector<float> g1, g2;      // the two RMSNorms' gamma
  Conv c1, c2, sc;
  bool shortcut = false;
};

struct Down {
  std::vector<Res> res;
  bool down = false;              // a stride-2 resample
  Conv resample;
  int ft = 1, fs = 1;             // the average-down shortcut's factors
};

}  // namespace

struct Magi2ImageEncoder::Impl {
  Conv conv1;
  std::vector<Down> downs;
  Res mid0, mid2;
  std::vector<float> attn_g;
  Conv qkv, proj;
  std::vector<float> head_g;
  Conv head, out;                 // head 3x3 to 96, conv1 1x1 96 -> 96
  std::size_t bytes = 0;
};

Magi2ImageEncoder::~Magi2ImageEncoder() = default;

namespace {

// y[N, cout] = x[N, K] w[cout, K]^T + b.
void
gemm_(const float* x, std::size_t N, const Conv& c, int K, float* y)
{
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)N, c.cout, K,
              1.0f, x, K, c.w.data(), K, 0.0f, y, c.cout);
  for (std::size_t r = 0; r < N; ++r) {
    float* yr = y + r * c.cout;
    for (int o = 0; o < c.cout; ++o) { yr[o] += c.b[(std::size_t)o]; }
  }
}

// A 3x3 conv: stride 1 padded 1 all round, or stride 2 padded right and
// bottom only (ZeroPad2d((0, 1, 0, 1)) then the conv). A 1x1 conv when
// c.k == 1.
Act
conv_(const Act& x, const Conv& c, int stride)
{
  Act y;
  if (c.k == 1) {
    y.H = x.H; y.W = x.W; y.C = c.cout;
    y.v.resize((std::size_t)y.H * y.W * y.C);
    gemm_(x.v.data(), (std::size_t)x.H * x.W, c, x.C, y.v.data());
    return y;
  }
  y.H = stride == 1 ? x.H : x.H / 2;
  y.W = stride == 1 ? x.W : x.W / 2;
  y.C = c.cout;
  y.v.resize((std::size_t)y.H * y.W * y.C);
  const int K = 9 * x.C;
  const int pad = stride == 1 ? 1 : 0;
  // Rows a chunk: im2col kept under ~64 MB.
  const std::size_t per =
      std::max<std::size_t>(1, ((std::size_t)64 << 20) / ((std::size_t)K * 4));
  std::vector<float> col(std::min<std::size_t>(per, (std::size_t)y.H * y.W) *
                         (std::size_t)K);
  const std::size_t total = (std::size_t)y.H * y.W;
  for (std::size_t r0 = 0; r0 < total; r0 += per) {
    const std::size_t n = std::min(per, total - r0);
    for (std::size_t r = 0; r < n; ++r) {
      const int oy = (int)((r0 + r) / y.W), ox = (int)((r0 + r) % y.W);
      float* row = col.data() + r * K;
      for (int ky = 0; ky < 3; ++ky) {
        for (int kx = 0; kx < 3; ++kx) {
          const int iy = oy * stride + ky - pad, ix = ox * stride + kx - pad;
          float* dst = row + (std::size_t)(ky * 3 + kx) * x.C;
          if (iy < 0 || ix < 0 || iy >= x.H || ix >= x.W) {
            std::memset(dst, 0, (std::size_t)x.C * 4);
          } else {
            std::memcpy(dst, x.px(iy, ix), (std::size_t)x.C * 4);
          }
        }
      }
    }
    gemm_(col.data(), n, c, K, y.v.data() + r0 * y.C);
  }
  return y;
}

// F.normalize over the channels x sqrt(C) x gamma, then SiLU if asked.
void
norm_act_(Act& x, const std::vector<float>& g, bool silu)
{
  const float sc = std::sqrt((float)x.C);
  for (int y = 0; y < x.H; ++y) {
    for (int xx = 0; xx < x.W; ++xx) {
      float* p = x.px(y, xx);
      double ss = 0.0;
      for (int c = 0; c < x.C; ++c) { ss += (double)p[c] * p[c]; }
      const float inv = 1.0f / std::max((float)std::sqrt(ss), 1e-12f);
      for (int c = 0; c < x.C; ++c) {
        float v = p[c] * inv * sc * g[(std::size_t)c];
        if (silu) { v = v / (1.0f + std::exp(-v)); }
        p[c] = v;
      }
    }
  }
}

Act
res_(const Act& x, const Res& r)
{
  Act h = x;
  norm_act_(h, r.g1, true);
  h = conv_(h, r.c1, 1);
  norm_act_(h, r.g2, true);
  h = conv_(h, r.c2, 1);
  Act s = r.shortcut ? conv_(x, r.sc, 1) : x;
  for (std::size_t i = 0; i < h.v.size(); ++i) { h.v[i] += s.v[i]; }
  return h;
}

// AvgDown3D on one frame: time padded to its factor with ZERO frames in
// front, then each output channel the mean of `group` consecutive entries
// of the (c, ts, hy, wx)-ordered list.
Act
avg_down_(const Act& x, int cout, int ft, int fs)
{
  Act y;
  y.H = x.H / fs; y.W = x.W / fs; y.C = cout;
  y.v.assign((std::size_t)y.H * y.W * y.C, 0.0f);
  const int factor = ft * fs * fs;
  const int group = x.C * factor / cout;
  for (int oy = 0; oy < y.H; ++oy) {
    for (int ox = 0; ox < y.W; ++ox) {
      float* o = y.px(oy, ox);
      for (int oc = 0; oc < cout; ++oc) {
        double acc = 0.0;
        for (int g = 0; g < group; ++g) {
          const int f = oc * group + g;
          const int c = f / factor, rem = f % factor;
          const int ts = rem / (fs * fs), hy = (rem / fs) % fs,
                    wx = rem % fs;
          // The real frame is the LAST of the padded ft.
          if (ts != ft - 1) { continue; }
          acc += x.px(oy * fs + hy, ox * fs + wx)[c];
        }
        o[oc] = (float)(acc / (double)group);
      }
    }
  }
  return y;
}

// The middle's single-head attention over the pixels.
Act
attn_(const Act& x, const Magi2ImageEncoder::Impl& p)
{
  Act h = x;
  norm_act_(h, p.attn_g, false);
  Act qkv = conv_(h, p.qkv, 1);
  const int N = x.H * x.W, C = x.C;
  std::vector<float> q((std::size_t)N * C), k((std::size_t)N * C),
      v((std::size_t)N * C);
  for (int n = 0; n < N; ++n) {
    const float* r = qkv.v.data() + (std::size_t)n * 3 * C;
    std::memcpy(q.data() + (std::size_t)n * C, r, (std::size_t)C * 4);
    std::memcpy(k.data() + (std::size_t)n * C, r + C, (std::size_t)C * 4);
    std::memcpy(v.data() + (std::size_t)n * C, r + 2 * C, (std::size_t)C * 4);
  }
  std::vector<float> s((std::size_t)N * N);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, C,
              1.0f / std::sqrt((float)C), q.data(), C, k.data(), C, 0.0f,
              s.data(), N);
  for (int i = 0; i < N; ++i) {
    float* r = s.data() + (std::size_t)i * N;
    float m = r[0];
    for (int j = 1; j < N; ++j) { m = std::max(m, r[j]); }
    double z = 0.0;
    for (int j = 0; j < N; ++j) {
      r[j] = std::exp(r[j] - m);
      z += r[j];
    }
    const float iz = (float)(1.0 / z);
    for (int j = 0; j < N; ++j) { r[j] *= iz; }
  }
  Act o;
  o.H = x.H; o.W = x.W; o.C = C;
  o.v.resize((std::size_t)N * C);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, C, N, 1.0f,
              s.data(), N, v.data(), C, 0.0f, o.v.data(), C);
  Act y = conv_(o, p.proj, 1);
  for (std::size_t i = 0; i < y.v.size(); ++i) { y.v[i] += x.v[i]; }
  return y;
}

}  // namespace

std::unique_ptr<Magi2ImageEncoder>
Magi2ImageEncoder::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                        std::string* err)
{
  auto fail = [&](const std::string& m) -> std::unique_ptr<Magi2ImageEncoder> {
    if (err != nullptr) { *err = "MAGI-2 image encoder: " + m; }
    return nullptr;
  };
  if (ws == nullptr || mc == nullptr) { return fail("nothing to read"); }
  std::unique_ptr<Magi2ImageEncoder> e(new Magi2ImageEncoder());
  e->_p = std::make_unique<Impl>();
  Impl& p = *e->_p;
  std::string bad;
  auto host = [&](const std::string& n) {
    std::vector<float> v;
    const auto* ti = ws->src().info(n);
    if (ti == nullptr || ti->dtype != "F32") {
      bad = n;
      return v;
    }
    SharedBuffer b = ws->read(n, mc, WeightSet::Residency::Copied);
    if (b.empty()) {
      bad = n;
      return v;
    }
    v.resize(b.byte_size() / 4);
    std::memcpy(v.data(), b.contents(), v.size() * 4);
    p.bytes += v.size() * 4;
    return v;
  };
  // A 3D conv's last temporal tap (or a 2D / 1x1 conv) as [cout, taps*cin].
  auto conv = [&](const std::string& n, Conv* c) {
    const auto* ti = ws->src().info(n + ".weight");
    if (ti == nullptr) {
      bad = n;
      return;
    }
    const std::vector<int64_t>& s = ti->shape;
    c->cout = (int)s[0];
    c->cin = (int)s[1];
    const int kt = s.size() == 5 ? (int)s[2] : 1;
    const int kh = s.size() >= 4 ? (int)s[s.size() - 2] : 1;
    const int kw = s.size() >= 4 ? (int)s[s.size() - 1] : 1;
    c->k = kh;
    const std::vector<float> w = host(n + ".weight");
    c->b = host(n + ".bias");
    if (w.empty() || (int)c->b.size() != c->cout) {
      bad = n;
      return;
    }
    const int taps = kh * kw;
    c->w.assign((std::size_t)c->cout * taps * c->cin, 0.0f);
    for (int o = 0; o < c->cout; ++o) {
      for (int i = 0; i < c->cin; ++i) {
        for (int t = 0; t < taps; ++t) {
          // [cout, cin, kt, kh, kw]: the last temporal tap.
          const std::size_t src =
              (((std::size_t)o * c->cin + i) * kt + (kt - 1)) * taps + t;
          c->w[((std::size_t)o * taps + t) * c->cin + i] = w[src];
        }
      }
    }
    p.bytes -= w.size() * 4;
    p.bytes += c->w.size() * 4;
  };
  auto res = [&](const std::string& n, Res* r) {
    r->g1 = host(n + ".residual.0.gamma");
    conv(n + ".residual.2", &r->c1);
    r->g2 = host(n + ".residual.3.gamma");
    conv(n + ".residual.6", &r->c2);
    r->shortcut = ws->src().info(n + ".shortcut.weight") != nullptr;
    if (r->shortcut) { conv(n + ".shortcut", &r->sc); }
  };
  conv("encoder.conv1", &p.conv1);
  // The release's Wan2_2_VAE: dim_mult [1, 2, 4, 4], temporal_downsample
  // [False, True, True]; a spatial downsample in all but the last block.
  const bool tdown[4] = {false, true, true, false};
  for (int i = 0; i < 4; ++i) {
    Down d;
    const std::string b = fmt("encoder.downsamples.{}.downsamples", i)();
    for (int j = 0; j < 2; ++j) {
      Res r;
      res(fmt("{}.{}", b, j)(), &r);
      d.res.push_back(std::move(r));
    }
    d.down = i < 3;
    if (d.down) { conv(b + ".2.resample.1", &d.resample); }
    d.ft = tdown[i] ? 2 : 1;
    d.fs = d.down ? 2 : 1;
    p.downs.push_back(std::move(d));
  }
  res("encoder.middle.0", &p.mid0);
  p.attn_g = host("encoder.middle.1.norm.gamma");
  conv("encoder.middle.1.to_qkv", &p.qkv);
  conv("encoder.middle.1.proj", &p.proj);
  res("encoder.middle.2", &p.mid2);
  p.head_g = host("encoder.head.0.gamma");
  conv("encoder.head.2", &p.head);
  conv("conv1", &p.out);
  if (!bad.empty()) { return fail("no usable '" + bad + "'"); }
  if (p.conv1.cin != 12 || p.out.cout != 96) {
    return fail("not the Wan2.2 VAE (12 patchified channels in, 2 x 48 out)");
  }
  return e;
}

std::size_t
Magi2ImageEncoder::weight_bytes() const noexcept
{
  return _p ? _p->bytes : 0;
}

std::size_t
Magi2ImageEncoder::held_bytes(const MetalLlamaWeights& src)
{
  std::size_t b = 0;
  for (const std::string& nm : src.tensor_names()) {
    const bool enc = nm.compare(0, 8, "encoder.") == 0 &&
                     nm.find("time_conv") == std::string::npos;
    if (!enc && nm.compare(0, 6, "conv1.") != 0) { continue; }
    const auto* ti = src.info(nm);
    if (ti == nullptr) { continue; }
    std::size_t n = 1;
    for (const auto d : ti->shape) { n *= (std::size_t)d; }
    // A 3D conv keeps its last temporal tap: [co, ci, kt, kh, kw] -> 1/kt.
    if (ti->shape.size() == 5 && ti->shape[2] > 0) {
      n /= (std::size_t)ti->shape[2];
    }
    b += n * 4;
  }
  return b;
}

std::size_t
Magi2ImageEncoder::working_bytes(int H, int W)
{
  const std::size_t H2 = (std::size_t)std::max(0, H / 32 * 32);
  const std::size_t W2 = (std::size_t)std::max(0, W / 32 * 32);
  if (H2 == 0 || W2 == 0) { return 0; }
  const std::size_t f = sizeof(float);
  // The floor's resize: the bytes, their resize and its floats.
  const std::size_t in = (std::size_t)3 * H * W + 3 * H2 * W2 * (1 + f);
  // The top level: 160 channels at a quarter of the pixels, five alive.
  const std::size_t top = 5 * (H2 / 2) * (W2 / 2) * 160 * f;
  const std::size_t col = (std::size_t)64 << 20;
  // The middle, 640 channels at 1 / 256 of the pixels: its activations,
  // q | k | v and its scores.
  const std::size_t N = H2 * W2 / 256;
  const std::size_t mid = (6 + 3 + 3) * N * 640 * f + N * N * f;
  return in + top + col + mid;
}

bool
Magi2ImageEncoder::encode(const float* rgb, int H, int W,
                          std::vector<float>* z, int* h, int* w,
                          std::string* err) const
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "MAGI-2 image encoder: " + m; }
    return false;
  };
  const Impl& p = *_p;
  const int H2 = H / 32 * 32, W2 = W / 32 * 32;
  if (H2 <= 0 || W2 <= 0) { return fail("a picture under 32 pixels"); }
  std::vector<float> img;
  const float* src = rgb;
  if (H2 != H || W2 != W) {
    // The release's floor: its bytes, LANCZOS (PIL), to the grid.
    std::vector<std::uint8_t> u8((std::size_t)3 * H * W);
    for (std::size_t i = 0; i < u8.size(); ++i) {
      const float v = (rgb[i] + 1.0f) * 0.5f * 255.0f;
      u8[i] = (std::uint8_t)std::clamp((int)std::lround(v), 0, 255);
    }
    std::vector<std::uint8_t> rs;
    if (!minimax_h3::resize_lanczos_planar_rgb(u8.data(), H, W, H2, W2,
                                               &rs)) {
      return fail("the resize to the 32-pixel grid");
    }
    img.resize(rs.size());
    for (std::size_t i = 0; i < rs.size(); ++i) {
      img[i] = (float)rs[i] / 255.0f * 2.0f - 1.0f;
    }
    src = img.data();
  }
  // patchify(2): channel c * 4 + wx * 2 + hy of pixel (2y + hy, 2x + wx).
  Act x;
  x.H = H2 / 2; x.W = W2 / 2; x.C = 12;
  x.v.resize((std::size_t)x.H * x.W * 12);
  for (int y = 0; y < x.H; ++y) {
    for (int xx = 0; xx < x.W; ++xx) {
      float* o = x.px(y, xx);
      for (int c = 0; c < 3; ++c) {
        for (int hy = 0; hy < 2; ++hy) {
          for (int wx = 0; wx < 2; ++wx) {
            o[c * 4 + wx * 2 + hy] =
                src[((std::size_t)c * H2 + 2 * y + hy) * W2 + 2 * xx + wx];
          }
        }
      }
    }
  }
  x = conv_(x, p.conv1, 1);
  for (const Down& d : p.downs) {
    const Act in = x;
    for (const Res& r : d.res) { x = res_(x, r); }
    if (d.down) { x = conv_(x, d.resample, 2); }
    const Act s = avg_down_(in, x.C, d.ft, d.fs);
    if (s.v.size() != x.v.size()) { return fail("a down block's shapes"); }
    for (std::size_t i = 0; i < x.v.size(); ++i) { x.v[i] += s.v[i]; }
  }
  x = res_(x, p.mid0);
  x = attn_(x, p);
  x = res_(x, p.mid2);
  norm_act_(x, p.head_g, true);
  x = conv_(x, p.head, 1);
  x = conv_(x, p.out, 1);
  // The mean (first 48 channels), whitened, channel-major.
  *h = x.H;
  *w = x.W;
  z->assign((std::size_t)48 * x.H * x.W, 0.0f);
  for (int y = 0; y < x.H; ++y) {
    for (int xx = 0; xx < x.W; ++xx) {
      const float* px = x.px(y, xx);
      for (int c = 0; c < 48; ++c) {
        (*z)[((std::size_t)c * x.H + y) * x.W + xx] =
            (px[c] - magi2::kLatentMean[c]) * (1.0f / magi2::kLatentStd[c]);
      }
    }
  }
  return true;
}

}  // namespace genai
}  // namespace vpipe
