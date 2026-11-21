#include "generative-models/deepseek-v41/metal-deepseek-v41-vision.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/image-ops.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <thread>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
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

float
bf16_f32_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::uint16_t
f32_bf16_(float f)
{
  std::uint32_t b;
  std::memcpy(&b, &f, 4);
  b += 0x7fffu + ((b >> 16) & 1u);
  return (std::uint16_t)(b >> 16);
}

float
elt_round_(float v, bool bf16)
{
  if (bf16) { return bf16_f32_(f32_bf16_(v)); }
  return (float)(_Float16)v;
}

// Python's round(): to nearest, ties to even.
long long
py_round_(double x)
{
  const double f = std::floor(x);
  const double d = x - f;
  if (d > 0.5) { return (long long)f + 1; }
  if (d < 0.5) { return (long long)f; }
  return ((long long)f % 2 == 0) ? (long long)f : (long long)f + 1;
}

// Pillow's 8-bit BICUBIC resize of planar u8 [3, sh, sw] -> [3, dh, dw]:
// Pillow's BICUBIC resize on 8 bits a channel (libImaging/Resample.c), as
// the release's ImageOps.pad runs it: each output pixel's taps from the
// filter (a = -0.5, support 2, widened by the scale when shrinking),
// normalized in double, rounded to integers of PRECISION_BITS = 22, summed
// as integers from a half unit and shifted back, clipped -- a horizontal
// pass, then a vertical one, a pass the size does not change skipped.
// Integers make it exact (Pillow's bytes) and the rows independent, so
// they split over threads.
struct PilTaps {
  std::vector<int> lo, n;       // each output's first input and its count
  std::vector<std::int32_t> k;  // [out][ksize]
  int ksize = 0;
};

double
pil_bicubic_(double x)
{
  constexpr double a = -0.5;
  x = std::fabs(x);
  if (x < 1.0) { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0; }
  if (x < 2.0) { return (((x - 5.0) * x + 8.0) * x - 4.0) * a; }
  return 0.0;
}

PilTaps
pil_taps_(int in_size, int out_size)
{
  constexpr int kPrecisionBits = 32 - 8 - 2;
  PilTaps t;
  const double scale = (double)in_size / (double)out_size;
  const double fscale = std::max(scale, 1.0);
  const double support = 2.0 * fscale;
  t.ksize = (int)std::ceil(support) * 2 + 1;
  t.lo.resize((std::size_t)out_size);
  t.n.resize((std::size_t)out_size);
  t.k.assign((std::size_t)out_size * t.ksize, 0);
  std::vector<double> w((std::size_t)t.ksize);
  const double ss = 1.0 / fscale;
  for (int xx = 0; xx < out_size; ++xx) {
    const double center = (xx + 0.5) * scale;
    int xmin = (int)(center - support + 0.5);
    if (xmin < 0) { xmin = 0; }
    int xmax = (int)(center + support + 0.5);
    if (xmax > in_size) { xmax = in_size; }
    xmax -= xmin;
    double ww = 0.0;
    for (int x = 0; x < xmax; ++x) {
      const double v = pil_bicubic_((x + xmin - center + 0.5) * ss);
      w[(std::size_t)x] = v;
      ww += v;
    }
    for (int x = 0; x < xmax; ++x) {
      double v = ww != 0.0 ? w[(std::size_t)x] / ww : w[(std::size_t)x];
      v *= (double)(1 << kPrecisionBits);
      t.k[(std::size_t)xx * t.ksize + x] =
          (std::int32_t)(v < 0 ? std::trunc(-0.5 + v) : std::trunc(0.5 + v));
    }
    t.lo[(std::size_t)xx] = xmin;
    t.n[(std::size_t)xx] = xmax;
  }
  return t;
}

void
resize_bicubic_u8_(const std::uint8_t* src, int sh, int sw,
                   std::uint8_t* dst, int dh, int dw)
{
  constexpr int kPrecisionBits = 32 - 8 - 2;
  auto clip8 = [](std::int32_t v) -> std::uint8_t {
    v >>= kPrecisionBits;
    return (std::uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
  };
  constexpr std::int32_t kHalf = 1 << (kPrecisionBits - 1);
  // (VPIPE_DSV41_VISION_THREADS=1: one thread -- the bench's check)
  const char* te = std::getenv("VPIPE_DSV41_VISION_THREADS");
  const int nt = te != nullptr ? std::max(1, std::atoi(te))
                               : (int)std::clamp<unsigned>(
                                     std::thread::hardware_concurrency(), 1u,
                                     8u);
  auto rows = [nt](int n, const auto& body) {
    if (nt == 1 || n < 64) { body(0, n); return; }
    std::vector<std::thread> ts;
    const int step = (n + nt - 1) / nt;
    for (int a0 = 0; a0 < n; a0 += step) {
      ts.emplace_back([&, a0] { body(a0, std::min(n, a0 + step)); });
    }
    for (auto& t : ts) { t.join(); }
  };
  // horizontal: [3][sh][sw] -> [3][sh][dw], skipped at the same width
  std::vector<std::uint8_t> mid;
  const std::uint8_t* h = src;
  if (dw != sw) {
    const PilTaps tx = pil_taps_(sw, dw);
    mid.resize((std::size_t)3 * sh * dw);
    rows(3 * sh, [&](int r0, int r1) {
      for (int r = r0; r < r1; ++r) {
        const std::uint8_t* in = src + (std::size_t)r * sw;
        std::uint8_t* out = mid.data() + (std::size_t)r * dw;
        for (int x = 0; x < dw; ++x) {
          const std::int32_t* k = &tx.k[(std::size_t)x * tx.ksize];
          const int lo = tx.lo[(std::size_t)x], n = tx.n[(std::size_t)x];
          std::int32_t acc = kHalf;
          for (int t = 0; t < n; ++t) {
            acc += (std::int32_t)in[lo + t] * k[t];
          }
          out[x] = clip8(acc);
        }
      }
    });
    h = mid.data();
  }
  // vertical: [3][sh][dw] -> [3][dh][dw], skipped at the same height
  if (dh == sh) {
    std::memcpy(dst, h, (std::size_t)3 * dh * dw);
    return;
  }
  const PilTaps ty = pil_taps_(sh, dh);
  rows(3 * dh, [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
      const int c = r / dh, y = r % dh;
      const std::uint8_t* in = h + (std::size_t)c * sh * dw;
      std::uint8_t* out = dst + (std::size_t)r * dw;
      const std::int32_t* k = &ty.k[(std::size_t)y * ty.ksize];
      const int lo = ty.lo[(std::size_t)y], n = ty.n[(std::size_t)y];
      for (int x = 0; x < dw; ++x) {
        std::int32_t acc = kHalf;
        for (int t = 0; t < n; ++t) {
          acc += (std::int32_t)in[(std::size_t)(lo + t) * dw + x] * k[t];
        }
        out[x] = clip8(acc);
      }
    }
  });
}

}  // namespace

void
MetalDeepseekV41Vision::resize_like_pillow(const std::uint8_t* src, int sh,
                                           int sw, std::uint8_t* dst, int dh,
                                           int dw)
{
  resize_bicubic_u8_(src, sh, sw, dst, dh, dw);
}

MetalDeepseekV41Vision::~MetalDeepseekV41Vision() = default;

MetalDeepseekV41Vision::Plan
MetalDeepseekV41Vision::plan(const DeepseekV41Config& c, int W, int H)
{
  // image_processor.plan_image_grid / safe_resize / solve_resize_ratio
  const int p = c.vis_patch, r = c.vis_downsample;
  long long width = W, height = H;
  if (c.vis_max_wh_ratio > 0 && width > height * c.vis_max_wh_ratio) {
    width = height * c.vis_max_wh_ratio;
  }
  if (width * height > 0 && width * height < c.vis_min_pixels) {
    const double ratio =
        std::sqrt((double)c.vis_min_pixels / (double)(width * height));
    width = (long long)((double)width * ratio);
    height = (long long)((double)height * ratio);
  }
  long long bw = (width + p - 1) / p * p, bh = (height + p - 1) / p * p;
  auto grid = [&](long long bh_, long long bw_, int* gh, int* gw) {
    *gh = (int)(((bh_ / p) + r - 1) / r);
    *gw = (int)(((bw_ / p) + r - 1) / r);
  };
  Plan out;
  grid(bh, bw, &out.llm_h, &out.llm_w);
  if (out.span() > c.vis_max_tokens) {
    const int mt = c.vis_max_tokens;
    const double rr = (double)height / (double)width;
    const double mw = std::sqrt((mt - 2) / rr + 0.25) - 0.5;
    const double mh = mw * rr;
    const long long cell = (long long)p * r;
    if (mw < 1.0) {
      bh = (mt - 2) / 2 * cell;
      bw = cell;
    } else if (mh < 1.0) {
      bh = cell;
      bw = (mt - 3) * cell;
    } else {
      const double beta =
          std::min(std::floor(mw) * (double)cell / (double)width,
                   std::floor(mh) * (double)cell / (double)height);
      bh = (long long)std::floor((double)height * beta / p) * p;
      bw = (long long)std::floor((double)width * beta / p) * p;
    }
    grid(bh, bw, &out.llm_h, &out.llm_w);
  }
  out.best_h = (int)bh;
  out.best_w = (int)bw;
  out.vit_h = (int)(bh / p);
  out.vit_w = (int)(bw / p);
  return out;
}

std::vector<float>
MetalDeepseekV41Vision::patches(const DeepseekV41Config& c,
                                const std::uint8_t* rgb, int H, int W,
                                Plan* plan_out, bool bf16)
{
  const Plan pl = plan(c, W, H);
  if (plan_out != nullptr) { *plan_out = pl; }
  const int bw = pl.best_w, bh = pl.best_h;
  // ImageOps.pad -> ImageOps.contain: fit the picture inside, aspect kept
  std::vector<std::uint8_t> canvas((std::size_t)3 * bh * bw, 127);
  int nw = bw, nh = bh;
  const double im = (double)W / (double)H, dest = (double)bw / (double)bh;
  if (im != dest) {
    if (im > dest) {
      nh = (int)py_round_((double)H / (double)W * bw);
    } else {
      nw = (int)py_round_((double)W / (double)H * bh);
    }
  }
  std::vector<std::uint8_t> fit;
  const std::uint8_t* src = rgb;
  if (nw != W || nh != H) {
    fit.resize((std::size_t)3 * nh * nw);
    resize_bicubic_u8_(rgb, H, W, fit.data(), nh, nw);
    src = fit.data();
  }
  int ox = 0, oy = 0;
  if (nw != bw) {
    ox = (int)py_round_((double)(bw - nw) * 0.5);
  } else if (nh != bh) {
    oy = (int)py_round_((double)(bh - nh) * 0.5);
  }
  for (int ch = 0; ch < 3; ++ch) {
    for (int y = 0; y < nh; ++y) {
      const int ty = y + oy;
      if (ty < 0 || ty >= bh) { continue; }
      for (int x = 0; x < nw; ++x) {
        const int tx = x + ox;
        if (tx < 0 || tx >= bw) { continue; }
        canvas[((std::size_t)ch * bh + ty) * bw + tx] =
            src[((std::size_t)ch * nh + y) * nw + x];
      }
    }
  }
  // (x / 255 - 0.5) / 0.5, rounded to the compute dtype; patch-major,
  // (channel, row, column) within a patch.
  const int p = c.vis_patch, PP = 3 * p * p;
  std::vector<float> out((std::size_t)pl.vit_h * pl.vit_w * PP);
  for (int py = 0; py < pl.vit_h; ++py) {
    for (int px = 0; px < pl.vit_w; ++px) {
      float* o = out.data() + ((std::size_t)py * pl.vit_w + px) * PP;
      for (int ch = 0; ch < 3; ++ch) {
        for (int y = 0; y < p; ++y) {
          for (int x = 0; x < p; ++x) {
            const float u = (float)canvas[((std::size_t)ch * bh + py * p + y) *
                                              bw + px * p + x];
            const float v = (u / 255.0f - 0.5f) / 0.5f;
            o[(ch * p + y) * p + x] = elt_round_(v, bf16);
          }
        }
      }
    }
  }
  return out;
}

std::unique_ptr<MetalDeepseekV41Vision>
MetalDeepseekV41Vision::load(const std::shared_ptr<WeightSet>& ws,
                             MetalCompute* mc, const DeepseekV41Config& cfg,
                             bool bf16, std::string* err)
{
  if (ws == nullptr || mc == nullptr || cfg.vis_layers <= 0) {
    if (err != nullptr) { *err = "no vision tower"; }
    return nullptr;
  }
  auto v = std::unique_ptr<MetalDeepseekV41Vision>(new MetalDeepseekV41Vision);
  v->_mc = mc;
  v->_cfg = cfg;
  v->_bf16 = bf16;
  if (!v->init_(ws, err)) { return nullptr; }
  return v;
}

bool
MetalDeepseekV41Vision::init_(const std::shared_ptr<WeightSet>& ws,
                              std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = "[deepseek-v4.1 vision] " + std::move(m); }
    return false;
  };
  const DeepseekV41Config& c = _cfg;
  if (c.vis_dim % c.vis_heads != 0 || c.vis_dim / c.vis_heads != 64) {
    return fail("the attention runs head dim 64");
  }
  _ws = ws;
  MetalCompute* mc = _mc;
  const char* sfx = _bf16 ? "_bf16" : "";
  _lib = mc->load_library(std::string("deepseek_v41") + sfx);
  _lib_dense = mc->load_library(std::string("dense_gemm") + sfx);
  _lib_attn = mc->load_library("attn_steel");
  _fn_rms = _lib.function("dsv41_rms_norm");
  _fn_rope = _lib.function("dsv41_vit_rope");
  _fn_add = _lib.function("dsv41_add");
  _fn_unfold = _lib.function("dsv41_unfold");
  _fn_gelu = _lib.function("dsv41_gelu");
  _fn_swiglu = _lib.function("dsv41_swiglu_clamp");
  _fn_gemm = _lib_dense.function("dense_gemm_t_f16");
  for (const auto* f : {&_fn_rms, &_fn_rope, &_fn_add, &_fn_unfold,
                        &_fn_gelu, &_fn_swiglu, &_fn_gemm}) {
    if (!f->valid()) { return fail("a kernel did not load"); }
  }
  WeightSet& wts = *ws;
  const auto& src = wts.src();
  const bool bf16 = _bf16;
  std::string bad;
  auto elt = [&](const std::string& nm,
                 std::vector<std::int64_t> sh) -> SharedBuffer {
    const auto* ti = src.info(nm);
    if (ti == nullptr || ti->dtype != "BF16" || ti->shape != sh) {
      if (bad.empty()) { bad = nm; }
      return {};
    }
    SharedBuffer raw = wts.read(nm, _mc, WeightSet::Residency::Copied);
    if (raw.empty() || bf16) { return raw; }
    const std::size_t n = raw.byte_size() / 2;
    SharedBuffer out = _mc->make_shared_buffer(n * 2);
    const auto* s = static_cast<const std::uint16_t*>(raw.contents());
    auto* d = static_cast<_Float16*>(out.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)bf16_f32_(s[i]); }
    return out;
  };
  const std::int64_t VD = c.vis_dim, VI = c.vis_inter, H = c.dim;
  const std::int64_t PP = 3LL * c.vis_patch * c.vis_patch;
  const std::int64_t r2 = (std::int64_t)c.vis_downsample * c.vis_downsample;
  _patch_w = elt("vision.patch_embed.proj.weight", {VD, PP});
  _patch_b = elt("vision.patch_embed.proj.bias", {VD});
  _norm = elt("vision.norm.weight", {VD});
  _blocks.resize((std::size_t)c.vis_layers);
  for (int b = 0; b < c.vis_layers; ++b) {
    Block& k = _blocks[(std::size_t)b];
    const std::string p = std::format("vision.blocks.{}.", b);
    k.norm1 = elt(p + "norm1.weight", {VD});
    k.norm2 = elt(p + "norm2.weight", {VD});
    k.wqkv = elt(p + "attn.wqkv.weight", {3 * VD, VD});
    k.bqkv = elt(p + "attn.wqkv.bias", {3 * VD});
    k.wo = elt(p + "attn.wo.weight", {VD, VD});
    k.bo = elt(p + "attn.wo.bias", {VD});
    // the release names the MLP `mlp`, oMLX's packs `ffn`
    const std::string mp =
        src.info(p + "ffn.w1.weight") != nullptr ? "ffn." : "mlp.";
    k.w1 = elt(p + mp + "w1.weight", {2 * VI, VD});
    k.w2 = elt(p + mp + "w2.weight", {VD, VI});
  }
  _al_w1 = elt("aligner.w1.weight", {H, VD * r2});
  _al_b1 = elt("aligner.w1.bias", {H});
  _al_w2 = elt("aligner.w2.weight", {H, H});
  _al_b2 = elt("aligner.w2.bias", {H});
  _img_start = elt("image_start", {H});
  _img_end = elt("image_end", {H});
  _img_newline = elt("image_newline", {H});
  if (!bad.empty()) {
    return fail("tensor " + bad + " is missing or misshapen");
  }
  metal_compute::FunctionConstants fc;
  fc.set_bool(200, false).set_bool(201, false).set_bool(300, false)
      .set_bool(301, false).set_bool(302, false);
  // The matrix cores where the GPU has them (VPIPE_DSV41_VISION_NAX=0: the
  // kernels below): the matmuls on matmul2d tiles, the attention on NAX.
  {
    const char* e = std::getenv("VPIPE_DSV41_VISION_NAX");
    if (mc->supports_matrix_cores() && !(e != nullptr && std::atoi(e) == 0)) {
      _lib_mma = mc->load_library(std::string("dense_gemm_mma") + sfx);
      _lib_attn_nax = mc->load_library("attn_steel_nax");
      _fn_mma = _lib_mma.function("dense_gemm_mma_t_n128_f16");
      // the bias added in the tile's f32 destination, rounded once (a
      // pass of its own after it cost ~1.17x the bf16 error)
      _fn_mma_bias = _lib_mma.function("dense_gemm_mma_t_n128_bias_f16");
      metal_compute::ComputeFunction a = _lib_attn_nax.function(
          bf16 ? "attn_steel_nax_h_bd64_bf16" : "attn_steel_nax_h_bd64", fc);
      _nax = _fn_mma.valid() && _fn_mma_bias.valid() && a.valid();
      if (_nax) { _fn_attn = std::move(a); }
    }
  }
  if (!_nax) {
    _fn_attn = _lib_attn.function(
        bf16 ? "attn_steel_h_bd64_bf16" : "attn_steel_h_bd64", fc);
  }
  if (!_fn_attn.valid()) { return fail("attn_steel_h_bd64 did not load"); }
  _attn_params = _mc->make_shared_buffer(sizeof(SteelAttnParams));
  if (_attn_params.empty()) { return fail("out of memory"); }
  return true;
}

std::size_t
MetalDeepseekV41Vision::encode_scratch_bytes(const DeepseekV41Config& c)
{
  // encode_patches' buffers at the most cells a span admits, each cell
  // r x r patches
  const std::size_t r = (std::size_t)c.vis_downsample;
  const std::size_t L = (std::size_t)std::max(c.vis_max_tokens, 1);
  const std::size_t n = L * r * r;
  const std::size_t VD = (std::size_t)c.vis_dim;
  const std::size_t VI = (std::size_t)c.vis_inter;
  const std::size_t PP = 3 * (std::size_t)c.vis_patch * c.vis_patch;
  const std::size_t H = (std::size_t)c.dim, hd = 64;
  const std::size_t elems = n * PP + n * VD * 7 + n * VI * 3 +
                            L * VD * r * r + 2 * L * H;
  return elems * 2 + n * (hd / 2) * 8;
}

std::size_t
MetalDeepseekV41Vision::resident_bytes() const
{
  std::size_t b = 0;
  for (const SharedBuffer* s :
       {&_patch_w, &_patch_b, &_norm, &_al_w1, &_al_b1, &_al_w2, &_al_b2,
        &_img_start, &_img_end, &_img_newline}) {
    b += s->byte_size();
  }
  for (const Block& k : _blocks) {
    for (const SharedBuffer* s : {&k.norm1, &k.norm2, &k.wqkv, &k.bqkv,
                                  &k.wo, &k.bo, &k.w1, &k.w2}) {
      b += s->byte_size();
    }
  }
  return b;
}

MetalDeepseekV41Vision::Result
MetalDeepseekV41Vision::encode(const std::uint8_t* rgb, int H, int W)
{
  Plan pl;
  const std::vector<float> px = patches(_cfg, rgb, H, W, &pl, _bf16);
  return encode_patches(px, pl);
}

MetalDeepseekV41Vision::Result
MetalDeepseekV41Vision::encode_patches(const std::vector<float>& px,
                                       const Plan& pl)
{
  Result res;
  // VPIPE_DSV41_VISION_PROFILE=1: where an encode's time goes, on stderr
  static const bool prof = std::getenv("VPIPE_DSV41_VISION_PROFILE") &&
                           std::atoi(std::getenv("VPIPE_DSV41_VISION_PROFILE"));
  using PClock = std::chrono::steady_clock;
  const auto p0 = PClock::now();
  const DeepseekV41Config& c = _cfg;
  const int n = pl.vit_h * pl.vit_w;
  const int VD = c.vis_dim, VI = c.vis_inter, heads = c.vis_heads, hd = 64;
  const int PP = 3 * c.vis_patch * c.vis_patch, H = c.dim;
  const int r = c.vis_downsample, rr = r * r;
  const int lh = (pl.vit_h + r - 1) / r, lw = (pl.vit_w + r - 1) / r;
  const int L = lh * lw;
  if (n <= 0 || px.size() != (std::size_t)n * PP || lh != pl.llm_h ||
      lw != pl.llm_w) {
    return res;
  }
  auto b = [&](std::size_t elems) {
    return _mc->make_shared_buffer(std::max<std::size_t>(elems * 2, 16));
  };
  SharedBuffer xin = b((std::size_t)n * PP), x = b((std::size_t)n * VD);
  SharedBuffer xn = b((std::size_t)n * VD), qkv = b((std::size_t)n * 3 * VD);
  SharedBuffer att = b((std::size_t)n * VD), proj = b((std::size_t)n * VD);
  SharedBuffer gu = b((std::size_t)n * 2 * VI), act = b((std::size_t)n * VI);
  SharedBuffer unf = b((std::size_t)L * VD * rr), h1 = b((std::size_t)L * H);
  SharedBuffer h2 = b((std::size_t)L * H);
  SharedBuffer tab = _mc->make_shared_buffer((std::size_t)n * (hd / 2) * 8);
  for (const SharedBuffer* s : {&xin, &x, &xn, &qkv, &att, &proj, &gu, &act,
                                &unf, &h1, &h2, &tab}) {
    if (s->empty()) { return res; }
  }
  {
    auto* d = static_cast<std::uint16_t*>(xin.contents());
    for (std::size_t i = 0; i < px.size(); ++i) {
      if (_bf16) {
        d[i] = f32_bf16_(px[i]);
      } else {
        const _Float16 h = (_Float16)px[i];
        std::memcpy(&d[i], &h, 2);
      }
    }
  }
  // get_vision_cos_sin: inv_freq over dim hd/2, the first half of the
  // pairs at the patch's row, the second at its column.
  {
    const int dim = hd / 2, nf = dim / 2;
    std::vector<float> inv((std::size_t)nf);
    for (int i = 0; i < nf; ++i) {
      inv[(std::size_t)i] = 1.0f / (float)std::pow(
          (double)c.vis_rope_theta, (double)((float)(2 * i) / (float)dim));
    }
    auto* t = static_cast<float*>(tab.contents());
    for (int py = 0; py < pl.vit_h; ++py) {
      for (int pw = 0; pw < pl.vit_w; ++pw) {
        const std::size_t i = (std::size_t)py * pl.vit_w + pw;
        for (int a = 0; a < 2; ++a) {
          for (int f = 0; f < nf; ++f) {
            const float ang = (float)(a == 0 ? py : pw) * inv[(std::size_t)f];
            const std::size_t o = (i * dim + (std::size_t)(a * nf + f)) * 2;
            t[o] = (float)std::cos((double)ang);
            t[o + 1] = (float)std::sin((double)ang);
          }
        }
      }
    }
  }
  auto* ap = static_cast<SteelAttnParams*>(_attn_params.contents());
  {
    // the kernel's tiles: NAX 64 x 32, the steel one 32 x 16
    const int bq = _nax ? 64 : 32, bk = _nax ? 32 : 16;
    ap->B = 1; ap->H = heads; ap->D = hd; ap->qL = n; ap->kL = n;
    ap->gqa_factor = 1;
    ap->scale = 1.0f / std::sqrt((float)hd);
    ap->NQ = (n + bq - 1) / bq; ap->NK = (n + bk - 1) / bk;
    ap->NQ_aligned = n / bq; ap->NK_aligned = n / bk;
    ap->qL_rem = n - ap->NQ_aligned * bq;
    ap->kL_rem = n - ap->NK_aligned * bk;
    ap->qL_off = 0;
    // q, k, v read in place from the fused rows [n, 3 * VD]
    for (std::int64_t* s : {ap->Q_strides, ap->K_strides, ap->V_strides}) {
      s[0] = (std::int64_t)n * 3 * VD;
      s[1] = hd;
      s[2] = 3 * VD;
    }
    ap->O_strides[0] = (std::int64_t)n * VD;
    ap->O_strides[1] = hd;
    ap->O_strides[2] = VD;
  }
  // one stream for the tower's life (a new queue's first commit waits)
  if (!_stream.valid()) { _stream = _mc->make_command_stream(); }
  metal_compute::CommandStream& stream = _stream;
  ComputeEncoder enc = stream.begin_compute();
  auto gemm = [&](const SharedBuffer& xa, const SharedBuffer& w,
                  const SharedBuffer* bias, const SharedBuffer& y, int M,
                  int N, int K) {
    if (_nax) {
      // matmul2d tiles of 128 x 128, the tails clamped (128 x 256 ran
      // 7-12% slower on these shapes); a bias in the tile's own variant
      constexpr int RN = 128;
      enc.set_function(bias != nullptr ? _fn_mma_bias : _fn_mma);
      enc.set_buffer(0, xa);
      enc.set_buffer(1, w);
      enc.set_buffer(2, bias != nullptr ? *bias : w);
      enc.set_buffer(3, y);
      enc.set_constant(4, K);
      enc.set_constant(5, N);
      enc.set_constant(6, M);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                    (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
      return;
    }
    enc.set_function(_fn_gemm);
    enc.set_buffer(0, xa);
    enc.set_buffer(1, w);
    enc.set_buffer(2, bias != nullptr ? *bias : w);
    enc.set_buffer(3, y);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    enc.set_constant(6, M);
    enc.set_constant(7, bias != nullptr ? 1 : 0);
    enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                  (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
  };
  auto rms = [&](const SharedBuffer& in, const SharedBuffer& w,
                 const SharedBuffer& out, int rows) {
    enc.set_function(_fn_rms);
    enc.set_buffer(0, in);
    enc.set_buffer(1, w);
    enc.set_buffer(2, out);
    enc.set_constant(3, VD);
    enc.set_constant(4, VD);
    enc.set_constant(5, VD);
    enc.set_constant(6, 1e-6f);
    enc.set_constant(7, 0);
    enc.dispatch({(unsigned)(256 * rows), 1, 1}, {256, 1, 1});
  };
  auto add = [&](const SharedBuffer& a, const SharedBuffer& d, int cnt) {
    enc.set_function(_fn_add);
    enc.set_buffer(0, a);
    enc.set_buffer(1, d);
    enc.set_constant(2, cnt);
    enc.dispatch({(unsigned)cnt, 1, 1}, {256, 1, 1});
  };
  gemm(xin, _patch_w, &_patch_b, x, n, VD, PP);
  for (const Block& k : _blocks) {
    rms(x, k.norm1, xn, n);
    gemm(xn, k.wqkv, &k.bqkv, qkv, n, 3 * VD, VD);
    enc.set_function(_fn_rope);
    enc.set_buffer(0, qkv);
    enc.set_buffer(1, tab);
    enc.set_constant(2, VD);
    enc.set_constant(3, hd);
    enc.set_constant(4, heads);
    enc.dispatch({(unsigned)(hd / 2), (unsigned)(2 * heads), (unsigned)n},
                 {(unsigned)(hd / 2), 1, 1});
    enc.set_function(_fn_attn);
    enc.set_buffer(0, qkv, 0);
    enc.set_buffer(1, qkv, (std::size_t)VD * 2);
    enc.set_buffer(2, qkv, (std::size_t)2 * VD * 2);
    enc.set_buffer(3, att);
    enc.set_buffer(4, _attn_params);
    enc.dispatch({32 * (unsigned)ap->NQ, 4 * (unsigned)heads, 1},
                 {32, 4, 1});
    gemm(att, k.wo, &k.bo, proj, n, VD, VD);
    add(x, proj, n * VD);
    rms(x, k.norm2, xn, n);
    gemm(xn, k.w1, nullptr, gu, n, 2 * VI, VD);
    enc.set_function(_fn_swiglu);
    enc.set_buffer(0, gu);
    enc.set_buffer(1, act);
    enc.set_constant(2, VI);
    enc.set_constant(3, 0.0f);
    enc.dispatch({(unsigned)VI, (unsigned)n, 1}, {256, 1, 1});
    gemm(act, k.w2, nullptr, proj, n, VD, VI);
    add(x, proj, n * VD);
  }
  rms(x, _norm, xn, n);
  // the aligner
  enc.set_function(_fn_unfold);
  enc.set_buffer(0, xn);
  enc.set_buffer(1, unf);
  enc.set_constant(2, pl.vit_h);
  enc.set_constant(3, pl.vit_w);
  enc.set_constant(4, VD);
  enc.set_constant(5, r);
  enc.dispatch({(unsigned)(VD * rr), (unsigned)L, 1}, {256, 1, 1});
  gemm(unf, _al_w1, &_al_b1, h1, L, H, VD * rr);
  enc.set_function(_fn_gelu);
  enc.set_buffer(0, h1);
  enc.set_constant(1, L * H);
  enc.dispatch({(unsigned)(L * H), 1, 1}, {256, 1, 1});
  gemm(h1, _al_w2, &_al_b2, h2, L, H, H);
  enc.end();
  const auto p1 = PClock::now();
  if (!stream.commit().wait_ok()) { return res; }
  const auto p2 = PClock::now();
  // the span: start, each cell row then a newline, end
  auto rowf = [&](const SharedBuffer& sb, std::size_t off, float* dst) {
    const auto* s = static_cast<const std::uint16_t*>(sb.contents()) + off;
    for (int d = 0; d < H; ++d) {
      dst[d] = _bf16 ? bf16_f32_(s[d])
                     : (float)reinterpret_cast<const _Float16*>(s)[d];
    }
  };
  res.grid_h = lh;
  res.grid_w = lw;
  res.n_tokens = pl.span();
  res.rows.resize((std::size_t)res.n_tokens * H);
  float* o = res.rows.data();
  rowf(_img_start, 0, o);
  o += H;
  for (int y = 0; y < lh; ++y) {
    for (int xw = 0; xw < lw; ++xw) {
      rowf(h2, ((std::size_t)y * lw + xw) * H, o);
      o += H;
    }
    rowf(_img_newline, 0, o);
    o += H;
  }
  rowf(_img_end, 0, o);
  if (prof) {
    const auto p3 = PClock::now();
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::fprintf(stderr, "[dsv41-vision] %d patches, %d cells: host %.1f ms, "
                 "gpu %.1f ms, span %.1f ms\n", n, L, ms(p0, p1), ms(p1, p2),
                 ms(p2, p3));
  }
  return res;
}

}  // namespace vpipe::genai
