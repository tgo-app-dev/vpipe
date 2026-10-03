#include "generative-models/yue2/metal-oobleck-decoder.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace vpipe::genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

constexpr const char* kKey = "oobleck-dec/f16|";

// The residual units' dilations: a dilated convolution has the shape of
// an undilated one, so these are the architecture's, not the checkpoint's.
constexpr int kDilations[3] = {1, 3, 9};

// The banded im2col's column budget, in elements: the widest stage would
// otherwise want a [T * 1920, 7 * 64] matrix per tile.
constexpr std::size_t kColCap = (std::size_t)16 << 20;

bool
read_f32_(WeightSet& ws, MetalCompute* mc, const std::string& nm,
          std::vector<float>* out)
{
  const auto* info = ws.src().info(nm);
  if (info == nullptr || info->dtype != "F32") { return false; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer raw = ws.read(nm, mc, WeightSet::Residency::Copied);
  if (raw.empty() || raw.byte_size() < n * 4) { return false; }
  out->assign(static_cast<const float*>(raw.contents()),
              static_cast<const float*>(raw.contents()) + n);
  return true;
}

}  // namespace

int
MetalOobleckDecoder::Config::hop() const
{
  int h = 1;
  for (int s : strides) { h *= s; }
  return h;
}

bool
MetalOobleckDecoder::config_from_dir(const std::string& dir, Config* out,
                                     std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  std::ifstream in(std::filesystem::path(dir) / "config.json");
  if (!in) { return fail("no config.json under " + dir); }
  FlexData fd;
  try {
    fd = FlexData::from_json(in);
  } catch (const std::exception& e) {
    return fail(std::string("config.json: ") + e.what());
  }
  if (!fd.is_object()) { return fail("config.json is not an object"); }
  auto o = fd.as_object();
  if (!o.contains("model_type") ||
      o.at("model_type").as_string() != "yue2_vae") {
    return fail("not a YuE2 VAE (model_type != \"yue2_vae\")");
  }
  Config c;
  c.sample_rate = o.contains("sample_rate")
                      ? (int)o.at("sample_rate").as_int(c.sample_rate)
                      : c.sample_rate;
  c.halo_frames = o.contains("decode_halo_frames")
                      ? (int)o.at("decode_halo_frames").as_int(c.halo_frames)
                      : c.halo_frames;
  c.core_frames = o.contains("decode_core_frames")
                      ? (int)o.at("decode_core_frames").as_int(c.core_frames)
                      : c.core_frames;
  if (o.contains("decoder_config")) {
    const FlexData dc = o.at("decoder_config");
    if (dc.is_object()) {
      auto d = dc.as_object();
      auto geti = [&](const char* k, int v) {
        return d.contains(k) ? (int)d.at(k).as_int(v) : v;
      };
      c.latent_dim   = geti("latent_dim", c.latent_dim);
      c.channels     = geti("channels", c.channels);
      c.out_channels = geti("out_channels", c.out_channels);
      if (d.contains("final_tanh")) {
        c.final_tanh = d.at("final_tanh").as_bool(false);
      }
      auto ints = [&](const char* k, std::vector<int>* dst) {
        if (!d.contains(k)) { return; }
        const FlexData a = d.at(k);
        if (!a.is_array()) { return; }
        dst->clear();
        for (const FlexData& v : a.as_array()) {
          dst->push_back((int)v.as_int());
        }
      };
      ints("c_mults", &c.c_mults);
      ints("strides", &c.strides);
      // The released decoder: vanilla SnakeBeta, no filter, no nearest
      // upsampling. Anything else is a different decoder.
      if ((d.contains("use_filter") && d.at("use_filter").as_bool(false)) ||
          (d.contains("antialias_activation") &&
           d.at("antialias_activation").as_bool(false)) ||
          (d.contains("use_nearest_upsample") &&
           d.at("use_nearest_upsample").as_bool(false)) ||
          (d.contains("use_snake") && !d.at("use_snake").as_bool(true))) {
        return fail("unsupported Oobleck decoder options");
      }
    }
  }
  if (c.c_mults.size() != c.strides.size() || c.c_mults.empty()) {
    return fail("decoder c_mults and strides disagree");
  }
  *out = c;
  return true;
}

bool
MetalOobleckDecoder::is_oobleck_dir(const std::string& dir)
{
  Config c;
  return config_from_dir(dir, &c, nullptr);
}

MetalOobleckDecoder::~MetalOobleckDecoder() = default;

std::unique_ptr<MetalOobleckDecoder>
MetalOobleckDecoder::load(const std::string& dir, MetalCompute* mc,
                          std::string* err)
{
  Config cfg;
  if (!config_from_dir(dir, &cfg, err)) { return nullptr; }
  return load(WeightSet::open(dir, nullptr), mc, cfg, err);
}

std::unique_ptr<MetalOobleckDecoder>
MetalOobleckDecoder::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                          const Config& cfg, std::string* err)
{
  if (ws == nullptr || mc == nullptr) {
    if (err != nullptr) { *err = "no checkpoint or no metal device"; }
    return nullptr;
  }
  std::unique_ptr<MetalOobleckDecoder> m(new MetalOobleckDecoder());
  if (!m->init_(ws, mc, cfg, err)) { return nullptr; }
  return m;
}

SharedBuffer
MetalOobleckDecoder::f16_(WeightSet& ws, const std::string& nm)
{
  if (ws.src().info(nm) == nullptr) { return {}; }
  return ws.derived(std::string(kKey) + nm, [&]() -> SharedBuffer {
    std::vector<float> v;
    if (!read_f32_(ws, _mc, nm, &v)) { return {}; }
    SharedBuffer out = _mc->make_shared_buffer(v.size() * 2);
    if (out.empty()) { return {}; }
    auto* d = static_cast<_Float16*>(out.contents());
    for (std::size_t i = 0; i < v.size(); ++i) { d[i] = (_Float16)v[i]; }
    return out;
  });
}

// Fold weight-norm, transpose into the GEMM layout and narrow to f16 in
// one pass. The norm is over every dimension but the FIRST, which for a
// ConvTranspose1d is the input width -- the contraction axis.
MetalOobleckDecoder::Conv
MetalOobleckDecoder::conv_(WeightSet& ws, const std::string& nm,
                           bool transposed, int dilation, int stride)
{
  Conv c;
  const bool wn = ws.src().info(nm + ".weight_v") != nullptr;
  const std::string wname = wn ? (nm + ".weight_v") : (nm + ".weight");
  const auto* info = ws.src().info(wname);
  if (info == nullptr || info->shape.size() != 3) { return c; }
  const int d0 = (int)info->shape[0];
  const int d1 = (int)info->shape[1];
  const int k  = (int)info->shape[2];
  c.cin = transposed ? d0 : d1;
  c.cout = transposed ? d1 : d0;
  c.k = k;
  c.dilation = dilation;
  c.stride = stride;
  // Oobleck pads an ordinary convolution "same" and a transposed one by
  // ceil(stride / 2) -- NOT (k - stride) / 2, which differs at stride 5.
  c.pad = transposed ? (stride + 1) / 2 : dilation * (k - 1) / 2;
  c.b = f16_(ws, nm + ".bias");                 // the last conv has none

  const std::string key =
      std::string(kKey) + (transposed ? "ct|" : "c|") + nm;
  c.w = ws.derived(key, [&]() -> SharedBuffer {
    std::vector<float> v, g;
    if (!read_f32_(ws, _mc, wname, &v)) { return {}; }
    if (wn && !read_f32_(ws, _mc, nm + ".weight_g", &g)) { return {}; }
    const std::size_t n = (std::size_t)d0 * d1 * k;
    if (v.size() < n) { return {}; }
    std::vector<float> scale((std::size_t)d0, 1.0f);
    if (wn) {
      if (g.size() < (std::size_t)d0) { return {}; }
      for (int i = 0; i < d0; ++i) {
        double sq = 0.0;
        const float* row = v.data() + (std::size_t)i * d1 * k;
        for (std::size_t j = 0; j < (std::size_t)d1 * k; ++j) {
          sq += (double)row[j] * (double)row[j];
        }
        const double nrm = std::sqrt(sq);
        scale[(std::size_t)i] =
            nrm > 0.0 ? (float)((double)g[(std::size_t)i] / nrm) : 0.0f;
      }
    }
    SharedBuffer out = _mc->make_shared_buffer(n * 2);
    if (out.empty()) { return {}; }
    auto* d = static_cast<_Float16*>(out.contents());
    if (!transposed) {
      // [cout, cin, k] -> [cout, k * cin], (k, cin) within a row.
      for (int co = 0; co < d0; ++co) {
        for (int ci = 0; ci < d1; ++ci) {
          for (int kk = 0; kk < k; ++kk) {
            d[((std::size_t)co * k + kk) * d1 + ci] =
                (_Float16)(v[((std::size_t)co * d1 + ci) * k + kk] *
                           scale[(std::size_t)co]);
          }
        }
      }
    } else {
      // [cin, cout, k] -> [k * cout, cin]: every tap's contribution side
      // by side, for col2im to fold.
      for (int ci = 0; ci < d0; ++ci) {
        for (int co = 0; co < d1; ++co) {
          for (int kk = 0; kk < k; ++kk) {
            d[((std::size_t)kk * d1 + co) * d0 + ci] =
                (_Float16)(v[((std::size_t)ci * d1 + co) * k + kk] *
                           scale[(std::size_t)ci]);
          }
        }
      }
    }
    return out;
  });
  return c;
}

MetalOobleckDecoder::Snake
MetalOobleckDecoder::snake_(WeightSet& ws, const std::string& nm)
{
  Snake s;
  std::vector<float> a, b;
  if (!read_f32_(ws, _mc, nm + ".alpha", &a) ||
      !read_f32_(ws, _mc, nm + ".beta", &b) || a.size() != b.size() ||
      a.empty()) {
    return s;
  }
  s.c = (int)a.size();
  // Log space: x + sin(e^a x)^2 / (e^b + 1e-9), both exponentials taken
  // once per channel here.
  s.ealpha = ws.derived(std::string(kKey) + "ea|" + nm,
                        [&]() -> SharedBuffer {
    SharedBuffer o = _mc->make_shared_buffer(a.size() * 4);
    if (o.empty()) { return {}; }
    auto* d = static_cast<float*>(o.contents());
    for (std::size_t i = 0; i < a.size(); ++i) { d[i] = std::exp(a[i]); }
    return o;
  });
  s.rbeta = ws.derived(std::string(kKey) + "rb|" + nm,
                       [&]() -> SharedBuffer {
    SharedBuffer o = _mc->make_shared_buffer(b.size() * 4);
    if (o.empty()) { return {}; }
    auto* d = static_cast<float*>(o.contents());
    for (std::size_t i = 0; i < b.size(); ++i) {
      d[i] = 1.0f / (std::exp(b[i]) + 1e-9f);
    }
    return o;
  });
  if (s.rbeta.empty()) { s.ealpha = {}; }
  return s;
}

bool
MetalOobleckDecoder::init_(const std::shared_ptr<WeightSet>& ws,
                           MetalCompute* mc, const Config& cfg,
                           std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  _ws = ws;
  _mc = mc;
  _cfg = cfg;
  WeightSet& w = *ws;

  // decoder.layers: [0] the input conv, [1..n] the upsample blocks from
  // the widest down, [n+1] the closing SnakeBeta, [n+2] the output conv.
  const int nb = (int)cfg.strides.size();
  _conv_in = conv_(w, "decoder.layers.0", false, 1, 1);
  if (_conv_in.empty()) { return fail("decoder.layers.0 missing"); }
  _blocks.resize((std::size_t)nb);
  for (int bi = 0; bi < nb; ++bi) {
    const std::string p = "decoder.layers." + std::to_string(bi + 1);
    Block& b = _blocks[(std::size_t)bi];
    b.rate = cfg.strides[(std::size_t)(nb - 1 - bi)];
    b.act = snake_(w, p + ".layers.0");
    b.up = conv_(w, p + ".layers.1", true, 1, b.rate);
    if (b.act.empty() || b.up.empty() || b.up.k != 2 * b.rate) {
      return fail("decoder block " + std::to_string(bi + 1) +
                  " is incomplete");
    }
    for (int j = 0; j < 3; ++j) {
      const std::string r = p + ".layers." + std::to_string(2 + j) +
                            ".layers.";
      ResUnit& u = b.res[j];
      u.a1 = snake_(w, r + "0");
      u.c1 = conv_(w, r + "1", false, kDilations[j], 1);
      u.a2 = snake_(w, r + "2");
      u.c2 = conv_(w, r + "3", false, 1, 1);
      if (u.a1.empty() || u.c1.empty() || u.a2.empty() || u.c2.empty()) {
        return fail("residual unit " + r + " is incomplete");
      }
    }
  }
  _act_out = snake_(w, "decoder.layers." + std::to_string(nb + 1));
  _conv_out = conv_(w, "decoder.layers." + std::to_string(nb + 2), false,
                    1, 1);
  if (_act_out.empty() || _conv_out.empty() ||
      _conv_out.cout != cfg.out_channels) {
    return fail("the decoder's output stage is incomplete");
  }

  _lib_gemm = mc->load_library("dense_gemm");
  _lib_elt  = mc->load_library("llm_elementwise");
  _lib_avae = mc->load_library("audio_vae_1d");
  _fn_gemm   = _lib_gemm.function("dense_gemm_t_bm64_f16");
  _fn_im2col = _lib_avae.function("im2col_1d_tc_f16");
  _fn_col2im = _lib_avae.function("col2im_1d_tc_f16");
  _fn_snake  = _lib_avae.function("snake_beta_f16");
  _fn_add    = _lib_elt.function("residual_add_f16");
  if (!_fn_gemm.valid() || !_fn_im2col.valid() || !_fn_col2im.valid() ||
      !_fn_snake.valid() || !_fn_add.valid()) {
    return fail("an Oobleck kernel did not resolve");
  }
  // Gated on the hardware: matmul2d EMULATES without matrix cores.
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_YUE2_VAE_NO_MMA2") == nullptr) {
    _lib_mma = mc->load_library("dense_gemm_mma");
    _fn_mma = _lib_mma.function("dense_gemm_mma_t_n128_f16");
    _fn_mma_deep = _lib_mma.function("dense_gemm_mma_t_n128x256_f16");
    _fn_bias_rows = _lib_elt.function("bias_add_rows_f16");
    _use_mma = _fn_mma.valid() && _fn_mma_deep.valid() &&
               _fn_bias_rows.valid();
  }
  return true;
}

int
MetalOobleckDecoder::natural_length(int frames) const
{
  if (frames <= 0) { return 0; }
  int L = frames;
  for (const Block& b : _blocks) {
    L = (L - 1) * b.rate - 2 * b.up.pad + b.up.k;
  }
  return L;
}

void
MetalOobleckDecoder::decode_cost(int frames, std::size_t* pcm,
                                 std::size_t* arena) const
{
  *pcm = (std::size_t)std::max(0, natural_length(frames)) *
         (std::size_t)_cfg.out_channels * 4;
  // A tile's widest activation is about hop * frames * channels at the
  // last stage; three of them, the pre-fold and the im2col band.
  const std::size_t tf = (std::size_t)(_core + 2 * _cfg.halo_frames);
  const std::size_t act =
      tf * (std::size_t)_cfg.hop() * (std::size_t)_cfg.channels * 2;
  *arena = 5 * act + kColCap * 2;
}

std::size_t
MetalOobleckDecoder::resident_bytes() const
{
  std::size_t b = _conv_in.w.byte_size() + _conv_out.w.byte_size();
  for (const Block& k : _blocks) {
    b += k.up.w.byte_size();
    for (const ResUnit& u : k.res) {
      b += u.c1.w.byte_size() + u.c2.w.byte_size();
    }
  }
  return b;
}

bool
MetalOobleckDecoder::ensure_scratch_(int frames)
{
  if (frames <= _tile_frames) { return true; }
  // The widest activation and pre-fold over every stage of this tile.
  std::size_t act = (std::size_t)frames *
                    (std::size_t)std::max(_conv_in.cout, _cfg.latent_dim);
  std::size_t pre = 0;
  int L = frames;
  for (const Block& b : _blocks) {
    act = std::max(act, (std::size_t)L * b.up.cin);
    pre = std::max(pre, (std::size_t)L * b.up.k * b.up.cout);
    L = (L - 1) * b.rate - 2 * b.up.pad + b.up.k;
    act = std::max(act, (std::size_t)L * b.up.cout);
  }
  act = std::max(act, (std::size_t)L * _conv_out.cout);
  _a = _mc->make_shared_buffer(act * 2);
  _b = _mc->make_shared_buffer(act * 2);
  _c = _mc->make_shared_buffer(act * 2);
  _pre = _mc->make_shared_buffer(pre * 2);
  if (_col.empty()) {
    _col_cap = kColCap;
    _col = _mc->make_shared_buffer(_col_cap * 2);
  }
  if (_a.empty() || _b.empty() || _c.empty() || _pre.empty() ||
      _col.empty()) {
    _tile_frames = 0;
    return false;
  }
  _tile_frames = frames;
  return true;
}

// The matrix-core route. The weights are already dense f16 in the [N, K]
// order matmul2d reads, so there is no dequant step -- and the shape gate
// is the whole design: the residual units' convolutions (M in the
// thousands, N 64..1024, K up to 7k) are compute-bound and go here, the
// upsampling tail (M in the tens of thousands, N of 64..256) would leave a
// 128-wide tile mostly idle and keeps the steel kernel.
bool
MetalOobleckDecoder::gemm_mma_(ComputeEncoder& enc, const SharedBuffer& x,
                               const SharedBuffer& w,
                               const SharedBuffer* bias,
                               const SharedBuffer& y, std::size_t ye, int M,
                               int N, int K)
{
  if (!_use_mma || M < _mma_min_m || N < _mma_min_n) { return false; }
  // Past 2^31 bytes on any operand the tiles stop storing, silently.
  if (M > mma_row_band(N, K)) { return false; }
  const bool wide = mma_use_wide_tile(N, K);
  const int RN = wide ? 256 : 128;
  enc.set_function(wide ? _fn_mma_deep : _fn_mma);
  enc.set_buffer(0, x);
  enc.set_buffer(1, w);
  enc.set_buffer(2, w);          // bias slot, unread (has_bias = 0)
  enc.set_buffer(3, y, ye * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
  if (bias != nullptr) {
    const std::uint32_t total = (std::uint32_t)((std::size_t)M * N);
    enc.set_function(_fn_bias_rows);
    enc.set_buffer(0, y, ye * 2);
    enc.set_buffer(1, *bias);
    enc.set_constant(2, N);
    enc.set_constant(3, total);
    enc.dispatch({total, 1, 1}, {256, 1, 1});
  }
  return true;
}

void
MetalOobleckDecoder::gemm_(ComputeEncoder& enc, const SharedBuffer& x,
                           const SharedBuffer& w, const SharedBuffer* bias,
                           const SharedBuffer& y, std::size_t ye, int M,
                           int N, int K)
{
  if (gemm_mma_(enc, x, w, bias, y, ye, M, N, K)) { return; }
  enc.set_function(_fn_gemm);
  enc.set_buffer(0, x);
  enc.set_buffer(1, w);
  enc.set_buffer(2, bias != nullptr ? *bias : w);
  enc.set_buffer(3, y, ye * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, bias != nullptr ? 1 : 0);
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
}

void
MetalOobleckDecoder::conv_run_(ComputeEncoder& enc, const Conv& c,
                               const SharedBuffer& in,
                               const SharedBuffer& out, int T)
{
  const SharedBuffer* bias = c.b.empty() ? nullptr : &c.b;
  if (c.k == 1) {
    gemm_(enc, in, c.w, bias, out, 0, T, c.cout, c.cin);
    return;
  }
  const std::size_t per_row = (std::size_t)c.k * c.cin;
  std::size_t band = std::max<std::size_t>(_col_cap / per_row, 1);
  band = std::min(band, (std::size_t)T);
  for (std::size_t r0 = 0; r0 < (std::size_t)T; r0 += band) {
    const int n = (int)std::min(band, (std::size_t)T - r0);
    enc.set_function(_fn_im2col);
    enc.set_buffer(0, in);
    enc.set_buffer(1, _col);
    enc.set_constant(2, T);
    enc.set_constant(3, c.cin);
    enc.set_constant(4, c.k);
    enc.set_constant(5, c.dilation);
    enc.set_constant(6, c.pad);
    enc.set_constant(7, 1);
    enc.set_constant(8, T);
    enc.set_constant(9, (int)r0);
    enc.set_constant(10, n);
    enc.dispatch({(unsigned)per_row, (unsigned)n, 1}, {64, 1, 1});
    gemm_(enc, _col, c.w, bias, out, r0 * (std::size_t)c.cout, n, c.cout,
          (int)per_row);
  }
}

void
MetalOobleckDecoder::snake_run_(ComputeEncoder& enc, const Snake& s,
                                const SharedBuffer& in,
                                const SharedBuffer& out, int T)
{
  const int C = s.c;
  const unsigned tx = (unsigned)std::min(C, 32);
  const unsigned ty = std::max(1u, 256u / tx);
  enc.set_function(_fn_snake);
  enc.set_buffer(0, in);
  enc.set_buffer(1, s.ealpha);
  enc.set_buffer(2, s.rbeta);
  enc.set_buffer(3, out);
  enc.set_constant(4, C);
  enc.set_constant(5, T * C);
  enc.dispatch({(unsigned)C, (unsigned)T, 1}, {tx, ty, 1});
}

bool
MetalOobleckDecoder::decode_tile_(const float* z, int frames, int f0, int f1,
                                  std::vector<float>* tile, std::string* err)
{
  const int T = f1 - f0;
  const int Z = _cfg.latent_dim;
  if (!ensure_scratch_(T)) {
    if (err != nullptr) { *err = "VAE scratch allocation failed"; }
    return false;
  }
  // Channel-major [Z, frames] -> time-major [T, Z] f16.
  {
    auto* d = static_cast<_Float16*>(_b.contents());
    for (int t = 0; t < T; ++t) {
      for (int ch = 0; ch < Z; ++ch) {
        d[(std::size_t)t * Z + ch] =
            (_Float16)z[(std::size_t)ch * frames + (std::size_t)(f0 + t)];
      }
    }
  }
  CommandStream s = _mc->make_command_stream();
  int L = T;
  {
    ComputeEncoder enc = s.begin_compute();
    conv_run_(enc, _conv_in, _b, _a, L);
    for (const Block& b : _blocks) {
      const Conv& u = b.up;
      snake_run_(enc, b.act, _a, _b, L);
      gemm_(enc, _b, u.w, nullptr, _pre, 0, L, u.k * u.cout, u.cin);
      const int Lo = (L - 1) * b.rate - 2 * u.pad + u.k;
      {
        const unsigned tx = (unsigned)std::min(u.cout, 32);
        const unsigned ty = std::max(1u, 256u / tx);
        enc.set_function(_fn_col2im);
        enc.set_buffer(0, _pre);
        enc.set_buffer(1, u.b.empty() ? _pre : u.b);
        enc.set_buffer(2, _a);
        enc.set_constant(3, L);
        enc.set_constant(4, u.cout);
        enc.set_constant(5, u.k);
        enc.set_constant(6, b.rate);
        enc.set_constant(7, u.pad);
        enc.set_constant(8, Lo);
        enc.set_constant(9, u.b.empty() ? 0 : 1);
        enc.dispatch({(unsigned)u.cout, (unsigned)Lo, 1}, {tx, ty, 1});
      }
      L = Lo;
      const int n = L * u.cout;
      for (const ResUnit& r : b.res) {
        snake_run_(enc, r.a1, _a, _b, L);
        conv_run_(enc, r.c1, _b, _c, L);
        snake_run_(enc, r.a2, _c, _b, L);
        conv_run_(enc, r.c2, _b, _c, L);
        enc.set_function(_fn_add);
        enc.set_buffer(0, _a);
        enc.set_buffer(1, _c);
        enc.set_buffer(2, _a);
        enc.set_constant(3, n);
        enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
      }
    }
    snake_run_(enc, _act_out, _a, _b, L);
    conv_run_(enc, _conv_out, _b, _c, L);
  }
  std::string gerr;
  if (!s.commit().wait_ok(&gerr)) {
    if (err != nullptr) {
      *err = gerr.empty() ? "VAE decode failed" : gerr;
    }
    return false;
  }
  const int C = _cfg.out_channels;
  tile->resize((std::size_t)L * C);
  const auto* src = static_cast<const _Float16*>(_c.contents());
  for (std::size_t i = 0; i < tile->size(); ++i) {
    float v = (float)src[i];
    if (_cfg.final_tanh) { v = std::tanh(v); }
    (*tile)[i] = v;
  }
  return true;
}

bool
MetalOobleckDecoder::decode(const float* z, int frames,
                            std::vector<float>* pcm, bool clamp,
                            const ProgressFn& progress, std::string* err)
{
  if (z == nullptr || pcm == nullptr || frames <= 0) {
    if (err != nullptr) { *err = "empty latent"; }
    return false;
  }
  const int hop = _cfg.hop();
  const int C = _cfg.out_channels;
  const int halo = std::max(_cfg.halo_frames, 0);
  const int total = natural_length(frames);
  pcm->assign((std::size_t)C * (std::size_t)total, 0.0f);
  const int tiles = (frames + _core - 1) / _core;
  std::vector<float> tile;
  for (int ti = 0; ti < tiles; ++ti) {
    const int start = ti * _core;
    const int end = std::min(frames, start + _core);
    const int left = std::max(0, start - halo);
    const int right = std::min(frames, end + halo);
    if (!decode_tile_(z, frames, left, right, &tile, err)) { return false; }
    // Keep the core: [start * hop, min(end * hop, total)) of the song is
    // [(start - left) * hop, ...) of the tile.
    const int out0 = start * hop;
    const int out1 = std::min(end * hop, total);
    const int crop = (start - left) * hop;
    const int tl = (int)(tile.size() / (std::size_t)C);
    if (crop + (out1 - out0) > tl) {
      if (err != nullptr) { *err = "a VAE tile did not cover its core"; }
      return false;
    }
    for (int t = out0; t < out1; ++t) {
      const std::size_t src = (std::size_t)(crop + t - out0) * C;
      for (int ch = 0; ch < C; ++ch) {
        float v = tile[src + (std::size_t)ch];
        if (clamp) { v = std::clamp(v, -1.0f, 1.0f); }
        (*pcm)[(std::size_t)ch * total + (std::size_t)t] = v;
      }
    }
    if (progress && !progress(ti + 1, tiles)) {
      if (err != nullptr) { err->clear(); }
      return false;
    }
  }
  return true;
}

}  // namespace vpipe::genai
