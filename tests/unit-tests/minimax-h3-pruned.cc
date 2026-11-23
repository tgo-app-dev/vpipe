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
#include "generative-models/minimax-h3/minimax-h3-scheduler.h"

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

// pi-H3 (Lakonik/pi-H3): a LoRA plus WHOLE-TENSOR OVERWRITES in the
// diffusers spelling -- every block's pruned AdaLN, one block's out /
// fc1 / fc2, the output AdaLN, norm and heads -- applied whole while the
// adapter is live, and a pi-Flow sampling contract read off its header
// and its name. Against the same DiT with those tensors PATCHED IN (the
// Comfy names, fc1's halves swapped as ComfyUI-piFlow's key map swaps
// them) and the LoRA in this model's own names.
TEST(minimax_h3_pruned, pi_flow_overwrites_are_the_patched_file)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Env noqmm("VPIPE_H3_NO_QMM_AUTOTUNE", "1");
  Env notile("VPIPE_H3_NO_ATTN_TILE_AUTOTUNE", "1");
  const int H = 256, A = 256, F = 512, TXT = 256, LAYERS = 2, REFINER = 1,
            GRID = 1025, D = 18 * H, FD = 2 * H, RANK = 4;
  const std::string P = "model.diffusion_model.";
  std::mt19937 rng(23);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto rnd = [&](std::size_t n, float sd, float mean = 0.0f) {
    std::vector<float> v(n);
    for (float& x : v) { x = mean + sd * nd(rng); }
    return v;
  };
  // F16, and `v` replaced by the values it holds.
  auto f16 = [](const std::string& nm, std::vector<std::int64_t> shape,
                std::vector<float>& v) {
    T t{nm, "F16", std::move(shape), std::vector<std::uint8_t>(v.size() * 2)};
    for (std::size_t i = 0; i < v.size(); ++i) {
      const _Float16 h = (_Float16)v[i];
      std::memcpy(&t.bytes[i * 2], &h, 2);
      v[i] = (float)h;
    }
    return t;
  };
  // The base DiT, by name: what the patched file replaces.
  std::map<std::string, T> base;
  auto put = [&](T t) { base[t.name] = std::move(t); };
  {
    std::vector<float> v;
    put(f32_("video_patch_proj.weight", {H, 96}, rnd(H * 96, 0.05f)));
    put(f32_("video_patch_proj.bias", {H}, rnd(H, 0.01f)));
    put(f32_("audio_patch_proj.weight", {H, 32}, rnd(H * 32, 0.05f)));
    put(f32_("audio_patch_proj.bias", {H}, rnd(H, 0.01f)));
    v = rnd((std::size_t)H * TXT, 0.03f);
    put(bf16_("condition_proj.weight", {H, TXT}, v));
    v = rnd(H, 0.01f);
    put(bf16_("condition_proj.bias", {H}, v));
    std::vector<float> inv(16);
    for (int i = 0; i < 16; ++i) {
      inv[(std::size_t)i] = 1.0f / std::pow(10000.0f, (float)(2 * i) / 32.0f);
    }
    put(f32_("rope.inv_freq", {16}, inv));
    for (int L = 0; L < LAYERS + REFINER; ++L) {
      const std::string b = L < LAYERS
          ? "blocks." + std::to_string(L) + "."
          : "token_refiner.blocks." + std::to_string(L - LAYERS) + ".";
      v = rnd(H, 0.02f, 1.0f); put(bf16_(b + "norm1.weight", {H}, v));
      v = rnd(H, 0.02f, 1.0f); put(bf16_(b + "norm2.weight", {H}, v));
      v = rnd(128, 0.02f, 1.0f); put(bf16_(b + "attn.q_norm.weight", {128}, v));
      v = rnd(128, 0.02f, 1.0f); put(bf16_(b + "attn.k_norm.weight", {128}, v));
      v = rnd((std::size_t)3 * A * H, 0.04f);
      put(bf16_(b + "attn.qkv_proj.weight", {3 * A, H}, v));
      v = rnd((std::size_t)H * A, 0.04f);
      put(bf16_(b + "attn.out_proj.weight", {H, A}, v));
      v = rnd((std::size_t)2 * F * H, 0.04f);
      put(bf16_(b + "mlp.fc1.weight", {2 * F, H}, v));
      v = rnd((std::size_t)H * F, 0.04f);
      put(bf16_(b + "mlp.fc2.weight", {H, F}, v));
      if (L < LAYERS) {
        put(f32_(b + "adaln_proj.linear.weight", {D, 8}, rnd(D * 8, 0.1f)));
        put(f32_(b + "adaln_proj.linear.bias", {D}, rnd(D, 0.05f)));
      }
    }
    v = rnd(H, 0.02f, 1.0f);
    put(bf16_("token_refiner.final_norm.weight", {H}, v));
    v = rnd(H, 0.02f, 1.0f);
    put(bf16_("final_layer.norm.weight", {H}, v));
    put(f32_("final_layer.video_out.weight", {96, H}, rnd(96 * H, 0.05f)));
    put(f32_("final_layer.video_out.bias", {96}, rnd(96, 0.01f)));
    put(f32_("final_layer.audio_out.weight", {32, H}, rnd(32 * H, 0.05f)));
    put(f32_("final_layer.audio_out.bias", {32}, rnd(32, 0.01f)));
    put(f32_("final_layer.adaln_proj.linear.weight", {FD, 8},
             rnd(FD * 8, 0.1f)));
    put(f32_("final_layer.adaln_proj.linear.bias", {FD}, rnd(FD, 0.05f)));
    put(f32_("adaln_t_table", {GRID, 8}, rnd(GRID * 8, 0.3f)));
  }
  auto write_dit = [&](const fs::path& p, const std::map<std::string, T>& m) {
    std::vector<T> ts;
    for (const auto& [nm, t] : m) {
      ts.push_back(t);
      ts.back().name = P + nm;
    }
    return write_st_(p, ts);
  };
  // The adapter (diffusers names) and the patched DiT beside it.
  std::vector<T> ad;        // pi-H3's spelling
  std::vector<T> ad_lora;   // its LoRA, this model's names
  std::map<std::string, T> patched = base;
  auto over = [&](const std::string& dname, const std::string& cname,
                  std::vector<std::int64_t> shape, float sd, bool swap) {
    std::size_t n = 1;
    for (auto d : shape) { n *= (std::size_t)d; }
    std::vector<float> v = rnd(n, sd);
    ad.push_back(f16(dname, shape, v));
    if (swap) {
      const std::size_t h = n / 2;
      std::rotate(v.begin(), v.begin() + (std::ptrdiff_t)h, v.end());
    }
    // The patched file stores it as the base did: F32 or BF16.
    const std::string dt = base.at(cname).dtype;
    patched[cname] = dt == "F32" ? f32_(cname, shape, v)
                                 : bf16_(cname, shape, v);
  };
  for (int L = 0; L < LAYERS; ++L) {
    const std::string d = "transformer_blocks." + std::to_string(L) + ".";
    const std::string c = "blocks." + std::to_string(L) + ".";
    over(d + "adaln_proj.linear.weight", c + "adaln_proj.linear.weight",
         {D, 8}, 0.1f, false);
  }
  // Block 1 fine-tuned whole, its AdaLN bias too (pi-H3's last four).
  over("transformer_blocks.1.adaln_proj.linear.bias",
       "blocks.1.adaln_proj.linear.bias", {D}, 0.05f, false);
  over("transformer_blocks.1.attn.to_out.0.weight", "blocks.1.attn.out_proj"
       ".weight", {H, A}, 0.04f, false);
  over("transformer_blocks.1.ff.net.0.proj.weight", "blocks.1.mlp.fc1.weight",
       {2 * F, H}, 0.04f, true);
  over("transformer_blocks.1.ff.net.2.weight", "blocks.1.mlp.fc2.weight",
       {H, F}, 0.04f, false);
  over("norm_out.linear.weight", "final_layer.adaln_proj.linear.weight",
       {FD, 8}, 0.1f, false);
  over("norm_out.linear.bias", "final_layer.adaln_proj.linear.bias", {FD},
       0.05f, false);
  over("proj_out.weight", "final_layer.video_out.weight", {96, H}, 0.05f,
       false);
  over("proj_out.bias", "final_layer.video_out.bias", {96}, 0.01f, false);
  over("audio_proj_out.weight", "final_layer.audio_out.weight", {32, H},
       0.05f, false);
  over("audio_proj_out.bias", "final_layer.audio_out.bias", {32}, 0.01f,
       false);
  // The output norm's weight sits near 1, as a norm's does.
  {
    std::vector<float> v = rnd(H, 0.02f, 1.0f);
    ad.push_back(f16("norm_out.norm.weight", {H}, v));
    patched["final_layer.norm.weight"] =
        bf16_("final_layer.norm.weight", {H}, v);
  }
  // Block 0's LoRA, both spellings: fc1's B halves swapped for this
  // model's [gate; value].
  auto lora = [&](const std::string& dmod, const std::string& cmod, int n,
                  int k, bool swap) {
    std::vector<float> a = rnd((std::size_t)RANK * k, 0.3f);
    std::vector<float> b = rnd((std::size_t)n * RANK, 0.3f);
    ad.push_back(f16(dmod + ".lora_A.weight", {RANK, k}, a));
    ad.push_back(f16(dmod + ".lora_B.weight", {n, RANK}, b));
    if (swap) {
      std::rotate(b.begin(), b.begin() + (std::ptrdiff_t)(b.size() / 2),
                  b.end());
    }
    ad_lora.push_back(bf16_(cmod + ".lora_A.weight", {RANK, k}, a));
    ad_lora.push_back(bf16_(cmod + ".lora_B.weight", {n, RANK}, b));
  };
  lora("transformer_blocks.0.attn.to_out.0", "blocks.0.attn.out_proj", H, A,
       false);
  lora("transformer_blocks.0.ff.net.0.proj", "blocks.0.mlp.fc1", 2 * F, H,
       true);
  lora("transformer_blocks.0.ff.net.2", "blocks.0.mlp.fc2", H, F, false);

  const std::map<std::string, std::string> policy = {
      {"policy_config", "{\"image\": {\"mode\": \"grid\", \"type\": \"DX\"}, "
                        "\"audio\": {\"mode\": \"grid\", \"type\": \"DX\"}}"}};
  const fs::path fdit = tmp_("pi-base.safetensors");
  const fs::path fpat = tmp_("pi-patched.safetensors");
  const fs::path dir = tmp_("pi_h3_4step_adapter");
  fs::create_directories(dir);
  const fs::path fad = dir / "diffusion_pytorch_model.safetensors";
  const fs::path flo = tmp_("pi-lora.safetensors");
  ASSERT_TRUE(write_dit(fdit, base));
  ASSERT_TRUE(write_dit(fpat, patched));
  ASSERT_TRUE(write_st_(fad, ad, policy));
  ASSERT_TRUE(write_st_(flo, ad_lora));
  // Refusals: an unplaced whole tensor; a DWD head without its
  // modulation; a step count the release states no grid for.
  // DWD: the same adapter with dwd_norm_out and a (stand-in) pixel head
  // tensor -- zero, so the features are the plain LayerNorm; and random.
  const fs::path fdz = tmp_("pi_h3_4step_dwd_zero.safetensors");
  const fs::path fdr = tmp_("pi_h3_4step_dwd_rand.safetensors");
  std::vector<float> dwd_w, dwd_b;
  {
    std::vector<float> hv = rnd(16, 0.01f);
    for (int k = 0; k < 2; ++k) {
      std::vector<T> ts = ad;
      std::vector<float> w = k == 0 ? std::vector<float>((std::size_t)FD * 8)
                                    : rnd((std::size_t)FD * 8, 0.2f);
      std::vector<float> b = k == 0 ? std::vector<float>((std::size_t)FD)
                                    : rnd((std::size_t)FD, 0.1f);
      ts.push_back(f16("dwd_norm_out.linear.weight", {FD, 8}, w));
      ts.push_back(f16("dwd_norm_out.linear.bias", {FD}, b));
      std::vector<float> h = hv;
      ts.push_back(f16("dwd_pixel_refiner.norm_out.weight", {16}, h));
      ASSERT_TRUE(write_st_(k == 0 ? fdz : fdr, ts, policy));
      if (k == 1) { dwd_w = w; dwd_b = b; }
    }
  }
  const fs::path fodd = tmp_("pi_h3_4step_odd.safetensors");
  const fs::path fdwd = tmp_("pi_h3_4step_dwd.safetensors");
  const fs::path f3 = tmp_("pi_h3_3step.safetensors");
  const fs::path fcy = tmp_("pi_h3_4step_comfy.safetensors");
  {
    std::vector<T> ts = ad;
    std::vector<float> v = rnd((std::size_t)A * H, 0.01f);
    ts.push_back(f16("transformer_blocks.0.attn.to_q.weight", {A, H}, v));
    ASSERT_TRUE(write_st_(fodd, ts, policy));
    ts = ad;
    v = rnd(16, 0.01f);
    ts.push_back(f16("dwd_pixel_refiner.norm_out.weight", {16}, v));
    ASSERT_TRUE(write_st_(fdwd, ts, policy));
    ASSERT_TRUE(write_st_(f3, ad, policy));
    // ComfyUI's spelling of the same (pi-H3's int8 ConvRot copies): the
    // LoRA would bind and the overwrites be dropped -- refused instead.
    std::vector<T> cy = ad_lora;
    v = rnd((std::size_t)D * 8, 0.1f);
    cy.push_back(f16("blocks.0.adaln_proj.linear.weight", {D, 8}, v));
    ASSERT_TRUE(write_st_(fcy, cy, policy));
  }

  H3::Config pc;
  std::string err;
  ASSERT_TRUE(H3::config_from_json(fdit.string(), pc, &err));
  ASSERT_TRUE(pc.pruned());
  const int latf = 2, lath = 8, latw = 8, naud = 8, ntext = 8;
  h3::PackedLayout lay;
  const std::vector<int> tags((std::size_t)ntext, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(tags, latf, lath, latw, naud,
                                        pc.patch_h, pc.patch_w,
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
      lay.video_indices.size() * (std::size_t)pc.video_patch_elems(), 0.017f);
  const metal_compute::SharedBuffer abuf = input(
      (std::size_t)lay.num_audio_rows * (std::size_t)pc.audio_channels,
      0.031f);
  const metal_compute::SharedBuffer tb =
      input((std::size_t)ntext * pc.text_dim, 0.005f);
  std::vector<float> uniq;
  std::vector<int> ridx;
  h3::build_row_timesteps(lay, 0.6f, 0.45f, 0.999f, &uniq, &ridx);
  struct Out {
    std::vector<std::uint16_t> v, a, f;
    std::vector<float> pts;
    bool loaded = false;
    int modules = 0;
    bool contract = false;
    std::vector<double> sigmas;
    std::size_t lora_bytes = 0;
  };
  auto run = [&](const fs::path& file, const std::vector<H3::LoraSpec>& lo,
                 bool stream, float live = -1.0f, bool dwd = false) -> Out {
    Out o;
    std::unique_ptr<H3> m;
    try {
      m = H3::load(file.string(), mc, pc, stream, lo);
    } catch (const std::exception& e) {
      std::printf("[minimax_h3_pruned] refused: %s\n", e.what());
    }
    if (m == nullptr) { return o; }
    o.loaded = true;
    o.modules = lo.empty() ? 0 : m->lora_modules(0);
    o.lora_bytes = m->lora_bytes();
    if (live >= 0.0f) { m->set_lora_scale(0, live); }
    if (const auto* sc = m->sampling_contract()) {
      o.contract = true;
      o.sigmas = sc->sigmas;
    }
    m->set_gemm_route(H3::GemmRoute::kSteelBm32);
    H3::Step st;
    st.video = &vb;
    st.audio = &abuf;
    st.text = &tb;
    st.layout = &lay;
    st.timesteps = &uniq;
    st.row_timestep_index = &ridx;
    st.dwd_features = dwd;
    if (dwd) { (void)m->time_embedding(uniq, nullptr, &o.pts); }
    std::string ferr;
    H3::Velocity out = m->forward(st, &ferr);
    if (out.empty()) {
      std::printf("[minimax_h3_pruned] forward: %s\n", ferr.c_str());
      return o;
    }
    const auto* p = static_cast<const std::uint16_t*>(out.video.contents());
    o.v.assign(p, p + out.video.byte_size() / 2);
    const auto* q = static_cast<const std::uint16_t*>(out.audio.contents());
    o.a.assign(q, q + out.audio.byte_size() / 2);
    if (dwd && !out.features.empty()) {
      const auto* f =
          static_cast<const std::uint16_t*>(out.features.contents());
      o.f.assign(f, f + out.features.byte_size() / 2);
    }
    return o;
  };
  const Out b0 = run(fdit, {}, false);
  const Out x1 = run(fdit, {{fad.string(), 1.0f}}, false);
  const Out x1s = run(fdit, {{fad.string(), 1.0f}}, true);
  const Out q1 = run(fpat, {{flo.string(), 1.0f}}, false);
  const Out xh = run(fdit, {{fad.string(), 0.5f}}, false);
  const Out qh = run(fpat, {{flo.string(), 0.5f}}, false);
  const Out x0 = run(fdit, {{fad.string(), 1.0f}}, false, 0.0f);
  const Out odd = run(fdit, {{fodd.string(), 1.0f}}, false);
  const Out dwd = run(fdit, {{fdwd.string(), 1.0f}}, false);
  const Out s3 = run(fdit, {{f3.string(), 1.0f}}, false);
  const Out cyo = run(fdit, {{fcy.string(), 1.0f}}, false);
  ASSERT_TRUE(b0.loaded && x1.loaded && q1.loaded && !b0.v.empty());
  const double moved = rel_(x1.v, b0.v);
  const double vs = rel_(x1.v, q1.v), vsa = rel_(x1.a, q1.a);
  const double half = rel_(xh.v, qh.v);
  std::printf("[minimax_h3_pruned] pi-H3's overwrites + LoRA: %d modules, "
              "against the patched file %.3g (audio %.3g; at strength 0.5 "
              "%.3g), the adapter moves the base %.3g; streamed %s; at 0 %s "
              "the base\n", x1.modules, vs, vsa, half, moved,
              x1s.v == x1.v ? "IDENTICAL" : "DIFFERENT",
              x0.v == b0.v ? "IDENTICAL to" : "DIFFERENT from");
  // The patched file holds the overwrites' exact values; the adapter adds
  // the AdaLN's as its difference from the trunk's in f32, one rounding
  // apart at most -- so agreement to well inside the adapter's effect.
  EXPECT_TRUE(moved > 0.1);
  EXPECT_TRUE(vs < 2e-3 && vsa < 2e-3 && half < 2e-3);
  EXPECT_TRUE(x1s.v == x1.v);
  EXPECT_TRUE(x0.v == b0.v);
  // Overwrites: 2 AdaLN weights + 1 bias, 3 projections, final AdaLN
  // weight + bias, norm, 4 head tensors = 13; LoRA 3.
  EXPECT_TRUE(x1.modules == 13 + 3);
  // The release's 4-step grid, and its schedules as the reference runs
  // them (torch float32): video at shift 12, audio at 3.
  const std::vector<float> g = H3::pi_flow_raw_grid(4);
  ASSERT_TRUE(x1.contract && g.size() == 5 && x1.sigmas.size() == 5);
  std::printf("[minimax_h3_pruned] pi-Flow 4-step raw grid %.6f %.6f %.6f "
              "%.6f %.6f\n", g[0], g[1], g[2], g[3], g[4]);
  MiniMaxH3Scheduler sv(12.0), sa(3.0);
  ASSERT_TRUE(sv.set_sigmas(g) && sa.set_sigmas(g));
  const float kV[] = {0.99654f, 0.96445f, 0.8993f, 0.69565f, 0.0f};
  const float kA[] = {0.9863f, 0.87151f, 0.69065f, 0.36364f, 0.0f};
  bool grid_ok = sv.sigmas().size() == 5 && sa.sigmas().size() == 5;
  for (int i = 0; grid_ok && i < 5; ++i) {
    grid_ok = std::fabs(sv.sigmas()[(std::size_t)i] - kV[i]) < 6e-6 &&
              std::fabs(sa.sigmas()[(std::size_t)i] - kA[i]) < 6e-6 &&
              (double)g[(std::size_t)i] == x1.sigmas[(std::size_t)i];
  }
  EXPECT_TRUE(grid_ok);
  const std::vector<float> g2 = H3::pi_flow_raw_grid(2);
  EXPECT_TRUE(g2.size() == 3 && H3::pi_flow_raw_grid(3).empty());
  std::printf("[minimax_h3_pruned] refused: an unplaced tensor %s, a DWD "
              "head with no modulation %s, a 3-step grid %s, ComfyUI's "
              "spelling %s\n",
              odd.loaded ? "LOADED" : "refused",
              dwd.loaded ? "LOADED" : "refused",
              s3.loaded ? "LOADED" : "refused",
              cyo.loaded ? "LOADED" : "refused");
  EXPECT_TRUE(!odd.loaded && !dwd.loaded && !s3.loaded && !cyo.loaded);
  // Booked as held: the overwrites, the AdaLN differences and the LoRA,
  // and the patch scratch once beside them.
  const auto fp = H3::lora_footprint(fad.string(), LAYERS, &pc);
  const std::size_t patch = (std::size_t)D * 9 * 4;
  std::printf("[minimax_h3_pruned] booked %zu = held %zu + patch %zu\n",
              fp.whole, x1.lora_bytes, patch);
  EXPECT_TRUE(fp.whole == x1.lora_bytes + patch);

  // DWD's features: the video rows' last hidden states, LayerNormed
  // without an affine, then (1 + scale) x + shift with [scale | shift] =
  // W silu(bf16(c)) + b at each row's timestep. Zero W and b give the
  // plain norm; the random arm is held to it through the host's formula.
  const Out dz = run(fdit, {{fdz.string(), 1.0f}}, false, -1.0f, true);
  const Out dr = run(fdit, {{fdr.string(), 1.0f}}, false, -1.0f, true);
  const Out dno = run(fdit, {{fad.string(), 1.0f}}, false, -1.0f, true);
  const std::size_t nv = lay.video_indices.size();
  ASSERT_TRUE(dz.loaded && dr.loaded && dz.f.size() == nv * H &&
              dr.f.size() == nv * H && dr.v == dz.v);
  if (dz.f.size() == nv * H && dr.f.size() == nv * H &&
      dr.pts.size() == uniq.size() * 8) {
    double mean_err = 0.0, var_err = 0.0;
    for (std::size_t r = 0; r < nv; ++r) {
      double m1 = 0.0, m2 = 0.0;
      for (int c = 0; c < H; ++c) {
        const double x = bf16f_(dz.f[r * H + (std::size_t)c]);
        m1 += x;
        m2 += x * x;
      }
      m1 /= H;
      mean_err = std::max(mean_err, std::fabs(m1));
      var_err = std::max(var_err, std::fabs(m2 / H - m1 * m1 - 1.0));
    }
    double d = 0.0, n = 0.0;
    for (std::size_t r = 0; r < nv; ++r) {
      const int ti = ridx[(std::size_t)lay.video_indices[r]];
      float sv[8];
      for (int k = 0; k < 8; ++k) {
        const float p = bf16f_(rne_(dr.pts[(std::size_t)ti * 8 + k]));
        sv[k] = bf16f_(rne_(p / (1.0f + std::exp(-p))));
      }
      for (int c = 0; c < H; ++c) {
        double sc = dwd_b[(std::size_t)c], sh = dwd_b[(std::size_t)(H + c)];
        for (int k = 0; k < 8; ++k) {
          sc += (double)dwd_w[(std::size_t)c * 8 + k] * sv[k];
          sh += (double)dwd_w[(std::size_t)(H + c) * 8 + k] * sv[k];
        }
        const double want =
            (1.0 + sc) * bf16f_(dz.f[r * H + (std::size_t)c]) + sh;
        const double got = bf16f_(dr.f[r * H + (std::size_t)c]);
        d += (got - want) * (got - want);
        n += want * want;
      }
    }
    const double rel = std::sqrt(d / std::max(n, 1e-30));
    std::printf("[minimax_h3_pruned] DWD features: the plain norm's rows "
                "mean <= %.2g, var - 1 <= %.2g; modulated %.3g from the "
                "host's formula; the velocity the adapter's own; a "
                "non-DWD adapter asked for them %s\n", mean_err, var_err,
                rel, dno.loaded && dno.v.empty() ? "REFUSED" : "ran");
    EXPECT_TRUE(mean_err < 2e-2 && var_err < 3e-2 && rel < 6e-3);
  }
  EXPECT_TRUE(dno.loaded && dno.v.empty());
  for (const fs::path& f :
       {fdit, fpat, fad, flo, fodd, fdwd, f3, fdz, fdr, fcy}) {
    fs::remove(f);
  }
  fs::remove(dir);
}
