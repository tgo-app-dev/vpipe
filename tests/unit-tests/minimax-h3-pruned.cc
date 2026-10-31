// minimax-h3-pruned.cc -- ComfyUI's PRUNED MiniMax-H3 (its "curve form":
// an `adaln_t_table` and 8-wide AdaLN projections, no timestep MLP)
// against the released model it was pruned from.
//
// A small synthetic released DiT R is written, and its EXACT pruning P:
// the table holds the top-8 principal coordinates of R's curve
// t -> silu(temb(t)) on the 1025-knot grid, V their directions and m the
// mean, and each AdaLN projection is W8 = W V, b8 = b + W m. P must then
// run as R does -- alone, and under what moves the AdaLN:
//
//   - a full-model adapter's AdaLN modules, ported onto the curve;
//   - an adapter on the timestep MLP, and a flow map, whose embedding
//     leaves the curve: projected onto it, the rest baked through R's
//     AdaLN (Config::time_mlp_dir = R);
//
// and a ComfyUI `.diff` / `.diff_b` patch must be the patched file, bit
// for bit, and nothing at strength 0.
//
// Self-contained: writes its own files, loads no checkpoint.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;

namespace {

namespace fs = std::filesystem;
namespace h3 = vpipe::genai::minimax_h3;
using H3 = MetalMiniMaxH3Transformer;

struct T {
  std::string               name;
  std::string               dtype;
  std::vector<std::int64_t> shape;
  std::vector<std::uint8_t> bytes;
};

bool
write_st_(const fs::path& p, const std::vector<T>& ts,
          const std::map<std::string, std::string>& meta = {})
{
  std::string hdr = "{";
  if (!meta.empty()) {
    hdr += "\"__metadata__\":{";
    bool first = true;
    for (const auto& [k, v] : meta) {
      std::string esc;
      for (char c : v) {
        if (c == '"' || c == '\\') { esc += '\\'; }
        esc += c;
      }
      hdr += (first ? "" : ",") + std::string("\"") + k + "\":\"" + esc +
             "\"";
      first = false;
    }
    hdr += "}";
    if (!ts.empty()) { hdr += ","; }
  }
  std::uint64_t off = 0;
  for (std::size_t i = 0; i < ts.size(); ++i) {
    std::string shp;
    for (std::size_t d = 0; d < ts[i].shape.size(); ++d) {
      shp += (d ? "," : "") + std::to_string(ts[i].shape[d]);
    }
    hdr += (i ? "," : "") + std::string("\"") + ts[i].name +
           "\":{\"dtype\":\"" + ts[i].dtype + "\",\"shape\":[" + shp +
           "],\"data_offsets\":[" + std::to_string(off) + "," +
           std::to_string(off + ts[i].bytes.size()) + "]}";
    off += ts[i].bytes.size();
  }
  hdr += "}";
  while (hdr.size() % 8 != 0) { hdr += ' '; }
  std::ofstream f(p, std::ios::binary);
  if (!f) { return false; }
  const std::uint64_t n = hdr.size();
  f.write(reinterpret_cast<const char*>(&n), 8);
  f.write(hdr.data(), (std::streamsize)hdr.size());
  for (const T& t : ts) {
    f.write(reinterpret_cast<const char*>(t.bytes.data()),
            (std::streamsize)t.bytes.size());
  }
  return f.good();
}

fs::path
tmp_(const std::string& name)
{
  return fs::temp_directory_path() /
         ("vpipe-h3-pruned-" + std::to_string(::getpid()) + "-" + name);
}

std::uint16_t
rne_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

float
bf16f_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

T
f32_(const std::string& nm, std::vector<std::int64_t> shape,
     const std::vector<float>& v)
{
  T t{nm, "F32", std::move(shape), std::vector<std::uint8_t>(v.size() * 4)};
  std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
  return t;
}

// BF16, and `v` replaced by the values it holds.
T
bf16_(const std::string& nm, std::vector<std::int64_t> shape,
      std::vector<float>& v)
{
  T t{nm, "BF16", std::move(shape), std::vector<std::uint8_t>(v.size() * 2)};
  for (std::size_t i = 0; i < v.size(); ++i) {
    const std::uint16_t b = rne_(v[i]);
    std::memcpy(&t.bytes[i * 2], &b, 2);
    v[i] = bf16f_(b);
  }
  return t;
}

// Symmetric eigenvectors by cyclic Jacobi: `a` [n, n] destroyed, `v`
// [n, n] the vectors as COLUMNS, `w` the values.
void
jacobi_(std::vector<double>& a, int n, std::vector<double>* v,
        std::vector<double>* w)
{
  v->assign((std::size_t)n * n, 0.0);
  for (int i = 0; i < n; ++i) { (*v)[(std::size_t)i * n + i] = 1.0; }
  auto A = [&](int i, int j) -> double& { return a[(std::size_t)i * n + j]; };
  auto V = [&](int i, int j) -> double& {
    return (*v)[(std::size_t)i * n + j];
  };
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) { off += A(i, j) * A(i, j); }
    }
    if (off < 1e-30) { break; }
    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        if (std::fabs(A(p, q)) < 1e-300) { continue; }
        const double th = (A(q, q) - A(p, p)) / (2.0 * A(p, q));
        const double t = (th >= 0 ? 1.0 : -1.0) /
                         (std::fabs(th) + std::sqrt(th * th + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
        for (int k = 0; k < n; ++k) {
          const double akp = A(k, p), akq = A(k, q);
          A(k, p) = c * akp - s * akq;
          A(k, q) = s * akp + c * akq;
        }
        for (int k = 0; k < n; ++k) {
          const double apk = A(p, k), aqk = A(q, k);
          A(p, k) = c * apk - s * aqk;
          A(q, k) = s * apk + c * aqk;
        }
        for (int k = 0; k < n; ++k) {
          const double vkp = V(k, p), vkq = V(k, q);
          V(k, p) = c * vkp - s * vkq;
          V(k, q) = s * vkp + c * vkq;
        }
      }
    }
  }
  w->resize((std::size_t)n);
  for (int i = 0; i < n; ++i) { (*w)[(std::size_t)i] = A(i, i); }
}

double
rel_(const std::vector<std::uint16_t>& x, const std::vector<std::uint16_t>& y)
{
  if (x.empty() || x.size() != y.size()) { return 1e9; }
  double d = 0.0, n = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double a = bf16f_(x[i]), b = bf16f_(y[i]);
    d += (a - b) * (a - b);
    n += b * b;
  }
  return std::sqrt(d / std::max(n, 1e-30));
}

// The two outputs' DIFFERENCE from a base, compared: how well P carries
// what an adapter does to R.
double
rel_delta_(const std::vector<std::uint16_t>& xp,
           const std::vector<std::uint16_t>& p,
           const std::vector<std::uint16_t>& xr,
           const std::vector<std::uint16_t>& r)
{
  if (xp.empty() || xp.size() != p.size() || xr.size() != r.size() ||
      xp.size() != xr.size()) {
    return 1e9;
  }
  double d = 0.0, n = 0.0;
  for (std::size_t i = 0; i < xp.size(); ++i) {
    const double dp = (double)bf16f_(xp[i]) - bf16f_(p[i]);
    const double dr = (double)bf16f_(xr[i]) - bf16f_(r[i]);
    d += (dp - dr) * (dp - dr);
    n += dr * dr;
  }
  return std::sqrt(d / std::max(n, 1e-30));
}

struct Env {
  const char* k;
  Env(const char* key, const char* v) : k(key) { ::setenv(key, v, 1); }
  ~Env() { ::unsetenv(k); }
};

}  // namespace

TEST(minimax_h3_pruned, runs_as_the_released_model_it_was_pruned_from)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Env noqmm("VPIPE_H3_NO_QMM_AUTOTUNE", "1");
  Env notile("VPIPE_H3_NO_ATTN_TILE_AUTOTUNE", "1");

  // hidden 256, 2 heads of 128, ffn 512, text 256; the released time MLP
  // 256 -> 64 -> 64; two main blocks and one refiner block.
  const int H = 256, A = 256, F = 512, TXT = 256, FREQ = 256, TH = 64,
            M = 64, LAYERS = 2, REFINER = 1, GRID = 1025, D = 18 * H,
            FD = 2 * H, RANK = 4;
  const std::string P = "model.diffusion_model.";
  std::mt19937 rng(17);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto rnd = [&](std::size_t n, float sd, float mean = 0.0f) {
    std::vector<float> v(n);
    for (float& x : v) { x = mean + sd * nd(rng); }
    return v;
  };

  std::vector<T> shared;   // both files
  std::vector<T> rel;      // R alone
  auto add = [&](T t) {
    t.name = P + t.name;
    shared.push_back(std::move(t));
  };
  {
    std::vector<float> v;
    add(f32_("video_patch_proj.weight", {H, 96}, rnd(H * 96, 0.05f)));
    add(f32_("video_patch_proj.bias", {H}, rnd(H, 0.01f)));
    add(f32_("audio_patch_proj.weight", {H, 32}, rnd(H * 32, 0.05f)));
    add(f32_("audio_patch_proj.bias", {H}, rnd(H, 0.01f)));
    v = rnd((std::size_t)H * TXT, 0.03f);
    add(bf16_("condition_proj.weight", {H, TXT}, v));
    v = rnd(H, 0.01f);
    add(bf16_("condition_proj.bias", {H}, v));
    std::vector<float> inv(16);
    for (int i = 0; i < 16; ++i) {
      inv[(std::size_t)i] = 1.0f / std::pow(10000.0f, (float)(2 * i) / 32.0f);
    }
    add(f32_("rope.inv_freq", {16}, inv));
    for (int L = 0; L < LAYERS + REFINER; ++L) {
      const std::string b = L < LAYERS
          ? "blocks." + std::to_string(L) + "."
          : "token_refiner.blocks." + std::to_string(L - LAYERS) + ".";
      v = rnd(H, 0.02f, 1.0f); add(bf16_(b + "norm1.weight", {H}, v));
      v = rnd(H, 0.02f, 1.0f); add(bf16_(b + "norm2.weight", {H}, v));
      v = rnd(128, 0.02f, 1.0f); add(bf16_(b + "attn.q_norm.weight", {128}, v));
      v = rnd(128, 0.02f, 1.0f); add(bf16_(b + "attn.k_norm.weight", {128}, v));
      v = rnd((std::size_t)3 * A * H, 0.04f);
      add(bf16_(b + "attn.qkv_proj.weight", {3 * A, H}, v));
      v = rnd((std::size_t)H * A, 0.04f);
      add(bf16_(b + "attn.out_proj.weight", {H, A}, v));
      v = rnd((std::size_t)2 * F * H, 0.04f);
      add(bf16_(b + "mlp.fc1.weight", {2 * F, H}, v));
      v = rnd((std::size_t)H * F, 0.04f);
      add(bf16_(b + "mlp.fc2.weight", {H, F}, v));
    }
    v = rnd(H, 0.02f, 1.0f);
    add(bf16_("token_refiner.final_norm.weight", {H}, v));
    v = rnd(H, 0.02f, 1.0f);
    add(bf16_("final_layer.norm.weight", {H}, v));
    add(f32_("final_layer.video_out.weight", {96, H}, rnd(96 * H, 0.05f)));
    add(f32_("final_layer.video_out.bias", {96}, rnd(96, 0.01f)));
    add(f32_("final_layer.audio_out.weight", {32, H}, rnd(32 * H, 0.05f)));
    add(f32_("final_layer.audio_out.bias", {32}, rnd(32, 0.01f)));
  }
  // R's timestep MLP (F32, as released) and AdaLN (BF16).
  const std::vector<float> w1 = rnd((std::size_t)TH * FREQ, 0.15f);
  const std::vector<float> b1 = rnd(TH, 0.2f);
  const std::vector<float> w2 = rnd((std::size_t)M * TH, 0.3f);
  const std::vector<float> b2 = rnd(M, 0.2f);
  rel.push_back(f32_(P + "time_embedder.proj_in.weight", {TH, FREQ}, w1));
  rel.push_back(f32_(P + "time_embedder.proj_in.bias", {TH}, b1));
  rel.push_back(f32_(P + "time_embedder.proj_out.weight", {M, TH}, w2));
  rel.push_back(f32_(P + "time_embedder.proj_out.bias", {M}, b2));
  std::vector<std::vector<float>> aw(LAYERS + 1), ab(LAYERS + 1);
  for (int L = 0; L <= LAYERS; ++L) {
    const int n = L < LAYERS ? D : FD;
    const std::string nm = L < LAYERS
        ? "blocks." + std::to_string(L) + ".adaln_proj.linear"
        : std::string("final_layer.adaln_proj.linear");
    aw[(std::size_t)L] = rnd((std::size_t)n * M, 0.08f);
    ab[(std::size_t)L] = rnd((std::size_t)n, 0.05f);
    rel.push_back(bf16_(P + nm + ".weight", {n, M}, aw[(std::size_t)L]));
    rel.push_back(bf16_(P + nm + ".bias", {n}, ab[(std::size_t)L]));
  }

  // R's curve on the grid, in double as the model's own MLP runs it.
  std::vector<double> S((std::size_t)GRID * M);
  for (int g = 0; g < GRID; ++g) {
    const double t = (double)g / (GRID - 1);
    std::vector<double> e(FREQ), h(TH);
    for (int i = 0; i < FREQ / 2; ++i) {
      const double fr = std::exp(-std::log(1e4) * i / (FREQ / 2));
      e[(std::size_t)i] = std::cos(t * fr);
      e[(std::size_t)(FREQ / 2 + i)] = std::sin(t * fr);
    }
    for (int o = 0; o < TH; ++o) {
      double a = b1[(std::size_t)o];
      for (int i = 0; i < FREQ; ++i) {
        a += (double)w1[(std::size_t)o * FREQ + i] * e[(std::size_t)i];
      }
      h[(std::size_t)o] = a / (1.0 + std::exp(-a));
    }
    for (int o = 0; o < M; ++o) {
      double a = b2[(std::size_t)o];
      for (int i = 0; i < TH; ++i) {
        a += (double)w2[(std::size_t)o * TH + i] * h[(std::size_t)i];
      }
      S[(std::size_t)g * M + o] = a / (1.0 + std::exp(-a));
    }
  }
  // Its principal directions.
  std::vector<double> mean((std::size_t)M, 0.0), cov((std::size_t)M * M, 0.0);
  for (int g = 0; g < GRID; ++g) {
    for (int j = 0; j < M; ++j) {
      mean[(std::size_t)j] += S[(std::size_t)g * M + j];
    }
  }
  for (double& x : mean) { x /= GRID; }
  for (int g = 0; g < GRID; ++g) {
    for (int a = 0; a < M; ++a) {
      const double ca = S[(std::size_t)g * M + a] - mean[(std::size_t)a];
      for (int b = 0; b < M; ++b) {
        cov[(std::size_t)a * M + b] +=
            ca * (S[(std::size_t)g * M + b] - mean[(std::size_t)b]);
      }
    }
  }
  std::vector<double> vec, val;
  jacobi_(cov, M, &vec, &val);
  std::vector<int> order((std::size_t)M);
  for (int i = 0; i < M; ++i) { order[(std::size_t)i] = i; }
  std::sort(order.begin(), order.end(), [&](int x, int y) {
    return val[(std::size_t)x] > val[(std::size_t)y];
  });
  double all = 0.0, rest = 0.0;
  for (int i = 0; i < M; ++i) {
    all += std::max(0.0, val[(std::size_t)order[(std::size_t)i]]);
    if (i >= 8) {
      rest += std::max(0.0, val[(std::size_t)order[(std::size_t)i]]);
    }
  }
  std::printf("[minimax_h3_pruned] the synthetic curve: %.3g of its energy "
              "outside 8 dimensions\n", rest / all);
  ASSERT_TRUE(rest / all < 1e-7);
  if (!(rest / all < 1e-7)) { return; }
  // V [M, 8], the table T = (S - m) V.
  std::vector<double> V((std::size_t)M * 8);
  for (int j = 0; j < M; ++j) {
    for (int k = 0; k < 8; ++k) {
      V[(std::size_t)j * 8 + k] =
          vec[(std::size_t)j * M + order[(std::size_t)k]];
    }
  }
  std::vector<float> table((std::size_t)GRID * 8);
  for (int g = 0; g < GRID; ++g) {
    for (int k = 0; k < 8; ++k) {
      double a = 0.0;
      for (int j = 0; j < M; ++j) {
        a += (S[(std::size_t)g * M + j] - mean[(std::size_t)j]) *
             V[(std::size_t)j * 8 + k];
      }
      table[(std::size_t)g * 8 + k] = (float)a;
    }
  }
  // P's AdaLN: W8 = W V, b8 = b + W m, F32.
  std::vector<std::vector<float>> pw(LAYERS + 1), pb(LAYERS + 1);
  for (int L = 0; L <= LAYERS; ++L) {
    const int n = L < LAYERS ? D : FD;
    const std::vector<float>& w = aw[(std::size_t)L];
    pw[(std::size_t)L].resize((std::size_t)n * 8);
    pb[(std::size_t)L].resize((std::size_t)n);
    for (int r = 0; r < n; ++r) {
      double bm = ab[(std::size_t)L][(std::size_t)r];
      for (int j = 0; j < M; ++j) {
        bm += (double)w[(std::size_t)r * M + j] * mean[(std::size_t)j];
      }
      pb[(std::size_t)L][(std::size_t)r] = (float)bm;
      for (int k = 0; k < 8; ++k) {
        double a = 0.0;
        for (int j = 0; j < M; ++j) {
          a += (double)w[(std::size_t)r * M + j] * V[(std::size_t)j * 8 + k];
        }
        pw[(std::size_t)L][(std::size_t)r * 8 + k] = (float)a;
      }
    }
  }
  auto pruned_file = [&](const fs::path& p,
                         const std::vector<std::vector<float>>& w,
                         const std::vector<std::vector<float>>& b) {
    std::vector<T> ts = shared;
    ts.push_back(f32_(P + "adaln_t_table", {GRID, 8}, table));
    for (int L = 0; L <= LAYERS; ++L) {
      const int n = L < LAYERS ? D : FD;
      const std::string nm = L < LAYERS
          ? "blocks." + std::to_string(L) + ".adaln_proj.linear"
          : std::string("final_layer.adaln_proj.linear");
      ts.push_back(f32_(P + nm + ".weight", {n, 8}, w[(std::size_t)L]));
      ts.push_back(f32_(P + nm + ".bias", {n}, b[(std::size_t)L]));
    }
    return write_st_(p, ts);
  };
  const fs::path fr = tmp_("released.safetensors");
  const fs::path fp = tmp_("pruned.safetensors");
  {
    std::vector<T> ts = shared;
    for (T& t : rel) { ts.push_back(t); }
    ASSERT_TRUE(write_st_(fr, ts));
  }
  ASSERT_TRUE(pruned_file(fp, pw, pb));
  // The same file with BARE names, as Comfy-Org's official pruned files
  // ship (no model prefix, no metadata): read as ComfyUI's, qkv flat.
  const fs::path fbare = tmp_("pruned-bare.safetensors");
  {
    std::vector<T> ts = shared;
    ts.push_back(f32_(P + "adaln_t_table", {GRID, 8}, table));
    for (int L = 0; L <= LAYERS; ++L) {
      const int n = L < LAYERS ? D : FD;
      const std::string nm = L < LAYERS
          ? "blocks." + std::to_string(L) + ".adaln_proj.linear"
          : std::string("final_layer.adaln_proj.linear");
      ts.push_back(f32_(P + nm + ".weight", {n, 8}, pw[(std::size_t)L]));
      ts.push_back(f32_(P + nm + ".bias", {n}, pb[(std::size_t)L]));
    }
    for (T& t : ts) { t.name = t.name.substr(P.size()); }
    ASSERT_TRUE(write_st_(fbare, ts));
  }

  // ---- the adapters ---------------------------------------------------
  auto factors = [&](std::vector<T>* ts, const std::string& mod, int n,
                     int k, float sa, float sb) {
    std::vector<float> a = rnd((std::size_t)RANK * k, sa);
    std::vector<float> b = rnd((std::size_t)n * RANK, sb);
    ts->push_back(bf16_(mod + ".lora_A.weight", {RANK, k}, a));
    ts->push_back(bf16_(mod + ".lora_B.weight", {n, RANK}, b));
  };
  // A full-model adapter on the AdaLN.
  const fs::path la = tmp_("lora-adaln.safetensors");
  {
    std::vector<T> ts;
    for (int L = 0; L < LAYERS; ++L) {
      factors(&ts, "blocks." + std::to_string(L) + ".adaln_proj.linear", D,
              M, 0.3f, 0.3f);
    }
    factors(&ts, "final_layer.adaln_proj.linear", FD, M, 0.3f, 0.3f);
    ASSERT_TRUE(write_st_(la, ts));
  }
  // One on the timestep MLP.
  const fs::path lt = tmp_("lora-time.safetensors");
  {
    std::vector<T> ts;
    factors(&ts, "time_embedder.proj_out", M, TH, 0.4f, 0.4f);
    ASSERT_TRUE(write_st_(lt, ts));
  }
  // A flow map: both embedders adapted, a header.
  const fs::path lf = tmp_("lora-flow.safetensors");
  {
    std::vector<T> ts;
    factors(&ts, "time_embedder.proj_in", TH, FREQ, 0.2f, 0.2f);
    factors(&ts, "time_embedder.proj_out", M, TH, 0.2f, 0.2f);
    factors(&ts, "endpoint_time_embedder.proj_in", TH, FREQ, 0.3f, 0.3f);
    factors(&ts, "endpoint_time_embedder.proj_out", M, TH, 0.3f, 0.3f);
    ASSERT_TRUE(write_st_(lf, ts,
                          {{"hyperflow", "true"},
                           {"hyperflow_gate", "0.25"},
                           {"hyperflow_sigmas", "[1.0, 0.5, 0.0]"}}));
  }
  // ComfyUI weight diffs, and the file they patch.
  const float ks = 0.7f;
  const fs::path ld = tmp_("lora-diff.safetensors");
  const fs::path fpd = tmp_("pruned-patched.safetensors");
  {
    std::vector<T> ts;
    std::vector<std::vector<float>> w2p = pw, b2p = pb;
    for (int L = 0; L <= LAYERS; ++L) {
      const int n = L < LAYERS ? D : FD;
      const std::string nm = L < LAYERS
          ? "blocks." + std::to_string(L) + ".adaln_proj.linear"
          : std::string("final_layer.adaln_proj.linear");
      const std::vector<float> dw = rnd((std::size_t)n * 8, 0.05f);
      const std::vector<float> db = rnd((std::size_t)n, 0.02f);
      ts.push_back(f32_("diffusion_model." + nm + ".diff", {n, 8}, dw));
      ts.push_back(f32_("diffusion_model." + nm + ".diff_b", {n}, db));
      for (std::size_t i = 0; i < dw.size(); ++i) {
        w2p[(std::size_t)L][i] = std::fma(ks, dw[i], pw[(std::size_t)L][i]);
      }
      for (std::size_t i = 0; i < db.size(); ++i) {
        b2p[(std::size_t)L][i] = std::fma(ks, db[i], pb[(std::size_t)L][i]);
      }
    }
    ASSERT_TRUE(write_st_(ld, ts));
    ASSERT_TRUE(pruned_file(fpd, w2p, b2p));
  }

  // ---- runs -----------------------------------------------------------
  H3::Config rc, pc;
  std::string err;
  ASSERT_TRUE(H3::config_from_json(fr.string(), rc, &err));
  ASSERT_TRUE(H3::config_from_json(fp.string(), pc, &err));
  EXPECT_TRUE(!rc.pruned() && pc.pruned() && pc.curve_grid == GRID &&
              pc.time_dim == 8 && rc.time_dim == M);
  const int latf = 2, lath = 8, latw = 8, naud = 8, ntext = 8;
  h3::PackedLayout lay;
  const std::vector<int> tags((std::size_t)ntext, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(tags, latf, lath, latw, naud,
                                        rc.patch_h, rc.patch_w,
                                        h3::kAudioChannels,
                                        {h3::Anchor::kFirst}, &lay));
  auto input = [&](std::size_t n, float k) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* p = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      p[i] = rne_(std::sin((float)i * k) * 0.5f);
    }
    return b;
  };
  const metal_compute::SharedBuffer vb = input(
      lay.video_indices.size() * (std::size_t)rc.video_patch_elems(), 0.017f);
  const metal_compute::SharedBuffer abuf = input(
      (std::size_t)lay.num_audio_rows * (std::size_t)rc.audio_channels,
      0.031f);
  const metal_compute::SharedBuffer tb =
      input((std::size_t)ntext * rc.text_dim, 0.005f);
  const float kV[] = {0.85f, 0.4f};
  const float kA[] = {0.75f, 0.35f};
  std::vector<std::vector<float>> uniqs(2), ends(2);
  std::vector<std::vector<int>> ridx(2);
  for (int i = 0; i < 2; ++i) {
    h3::build_row_timesteps(lay, kV[i], kA[i], 1.0f, &uniqs[(std::size_t)i],
                            &ridx[(std::size_t)i]);
    for (float t : uniqs[(std::size_t)i]) {
      ends[(std::size_t)i].push_back(std::min(1.0f, t + 0.15f));
    }
  }
  struct Out {
    std::vector<std::uint16_t> v;
    int modules = 0;
    bool loaded = false;
  };
  // Two forwards (the second compared), baked over both when asked --
  // which a pruned DiT under a moved embedding requires.
  auto run = [&](const fs::path& file, H3::Config cfg,
                 const std::vector<H3::LoraSpec>& loras, bool bake,
                 bool flow, float live = -1.0f) -> Out {
    Out o;
    std::unique_ptr<H3> m;
    try {
      m = H3::load(file.string(), mc, cfg, false, loras);
    } catch (const std::exception& e) {
      std::printf("[minimax_h3_pruned] refused: %s\n", e.what());
    }
    if (m == nullptr) { return o; }
    o.loaded = true;
    o.modules = loras.empty() ? 0 : m->lora_modules(0);
    if (live >= 0.0f) { m->set_lora_scale(0, live); }
    m->set_gemm_route(H3::GemmRoute::kSteelBm32);
    std::string berr;
    if (bake && !m->bake_adaln(uniqs, &berr, flow ? &ends : nullptr)) {
      std::printf("[minimax_h3_pruned] bake: %s\n", berr.c_str());
      return o;
    }
    H3::Velocity out;
    for (int i = 0; i < 2; ++i) {
      H3::Step st;
      st.video = &vb;
      st.audio = &abuf;
      st.text = &tb;
      st.layout = &lay;
      st.timesteps = &uniqs[(std::size_t)i];
      st.row_timestep_index = &ridx[(std::size_t)i];
      if (flow) { st.endpoints = &ends[(std::size_t)i]; }
      if (bake) { st.schedule_index = i; }
      std::string ferr;
      out = m->forward(st, &ferr);
      if (out.empty()) {
        std::printf("[minimax_h3_pruned] forward: %s\n", ferr.c_str());
        return o;
      }
    }
    const auto* p = static_cast<const std::uint16_t*>(out.video.contents());
    o.v.assign(p, p + out.video.byte_size() / 2);
    return o;
  };
  H3::Config pcm = pc;
  pcm.time_mlp_dir = fr.string();

  const Out r0 = run(fr, rc, {}, false, false);
  const Out p0 = run(fp, pcm, {}, false, false);
  H3::Config bc;
  const bool bare_read = H3::config_from_json(fbare.string(), bc, &err) &&
                         bc.pruned() && !bc.qkv_per_head;
  const Out pbare = bare_read ? run(fbare, bc, {}, false, false) : Out{};
  ASSERT_TRUE(!r0.v.empty() && !p0.v.empty());
  const double base = rel_(p0.v, r0.v);

  // A full-model AdaLN adapter: R binds factors, P ports them.
  const Out ra = run(fr, rc, {{la.string(), 1.0f}}, false, false);
  const Out pa = run(fp, pcm, {{la.string(), 1.0f}}, false, false);
  // On the timestep MLP, and a flow map: P projects and bakes the rest.
  // At strength 0.25: at 1 this adapter drives the synthetic forward
  // where a 0.4% change of its strength moves R 1.1%, 3.3x the linear
  // share -- and P against R grows with it (8.6%), the two models'
  // rounding amplified rather than any part missing; it falls to the
  // pruning's own 1% as the strength does (3.1 / 1.5 / 1.0% at 0.5 /
  // 0.25 / 0.1).
  const float kts = 0.25f;
  const Out rt = run(fr, rc, {{lt.string(), kts}}, true, false);
  const Out pt = run(fp, pcm, {{lt.string(), kts}}, true, false);
  const Out rf = run(fr, rc, {{lf.string(), 1.0f}}, true, true);
  const Out pf = run(fp, pcm, {{lf.string(), 1.0f}}, true, true);
  // Without the released MLP an adapter of the timestep MLP alone binds
  // nothing and is refused, and so is a flow map.
  const Out pt_none = run(fp, pc, {{lt.string(), 1.0f}}, false, false);
  const Out pf_none = run(fp, pc, {{lf.string(), 1.0f}}, true, true);
  // A ComfyUI patch, against the patched file; and at strength 0.
  const Out pd = run(fp, pc, {{ld.string(), ks}}, false, false);
  const Out pdt = run(fpd, pc, {}, false, false);
  const Out pd0 = run(fp, pc, {{ld.string(), ks}}, false, false, 0.0f);

  const double da = rel_delta_(pa.v, p0.v, ra.v, r0.v);
  const double dt = rel_(pt.v, rt.v);
  const double df = rel_(pf.v, rf.v);
  std::printf("[minimax_h3_pruned] P against R: alone %.3g; under a "
              "full-model AdaLN adapter, its effect carried to %.3g (it "
              "moves R %.3g); under a timestep-MLP adapter %.3g (it moves "
              "R %.3g); under a flow map %.3g (it moves R %.3g)\n",
              base, da, rel_(ra.v, r0.v), dt, rel_(rt.v, r0.v), df,
              rel_(rf.v, r0.v));
  std::printf("[minimax_h3_pruned] ComfyUI .diff: %s the patched file, %s "
              "the base at strength 0 (%d module(s)); with no released "
              "MLP the timestep adapter %s, the flow map %s\n",
              pd.v == pdt.v ? "IDENTICAL to" : "DIFFERENT from",
              pd0.v == p0.v ? "IDENTICAL to" : "DIFFERENT from", pd.modules,
              pt_none.loaded ? "LOADED" : "refused",
              pf_none.loaded ? "LOADED" : "refused");
  // The pruning is exact up to the curve's 8-dimensional fit and the
  // forward's rounding; each adapter must move R well past that, and P
  // must carry the move to within a few percent of it.
  EXPECT_TRUE(base < 2e-2);
  EXPECT_TRUE(rel_(ra.v, r0.v) > 5.0 * base && da < 5e-2);
  EXPECT_TRUE(rel_(rt.v, r0.v) > 5.0 * base && dt < 2.5 * base);
  EXPECT_TRUE(rel_(rf.v, r0.v) > 5.0 * base &&
              df < 0.1 * rel_(rf.v, r0.v));
  EXPECT_TRUE(pd.modules > 0 && !pd.v.empty() && pd.v == pdt.v);
  EXPECT_TRUE(pd0.v == p0.v);
  EXPECT_TRUE(!pt_none.loaded && !pf_none.loaded);
  std::printf("[minimax_h3_pruned] bare names (Comfy-Org's official form): "
              "%s, %s the prefixed file\n", bare_read ? "read flat" : "REFUSED",
              pbare.v == p0.v ? "IDENTICAL to" : "DIFFERENT from");
  EXPECT_TRUE(bare_read && pbare.v == p0.v);
  for (const fs::path& f : {fr, fp, fbare, la, lt, lf, ld, fpd}) {
    fs::remove(f);
  }
}
