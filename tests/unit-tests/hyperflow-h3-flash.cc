// hyperflow-h3-flash.cc -- HyperFlow H3 Flash (VideoRebirthAI/hyperflow-
// h3-flash), its ComfyUI single file, against the plain H3 it is made of.
//
// A small synthetic released DiT R of four blocks is written, and a
// Flash file F of the same model as the release ships it:
//
//   - block 2 LEFT OUT and the rest renumbered 0..2, the `hyperflow`
//     metadata naming the release's index dropped;
//   - LOW-RANK projections -- block 0 all of them (q / k / v pairs of
//     three different ranks beside one another), block 2 its fc2 alone --
//     as `<proj>.down.weight` / `.up.weight`;
//   - its own ENDPOINT embedder beside the base one, and its gate.
//
// The reference is R3, the released model with block 2 removed and every
// low-rank pair multiplied out into its dense weight, under a FLOW-MAP
// ADAPTER (the HyperFlow LoRA form) whose two embedders, base plus
// factors, are F's two embedders: F must run as R3 under it. Then an
// adapter named by the RELEASE's block indices must reach the same
// blocks on F as its renumbered twin on R3, its module on the dropped
// block skipped. Streamed and preloaded alike.
//
// Self-contained: writes its own files, loads no checkpoint.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/tensor-parallel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <exception>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
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
         ("vpipe-h3-flash-" + std::to_string(::getpid()) + "-" + name);
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

// TWO RANKS IN ONE PROCESS: each exchange leaves this rank's partial
// under its sequence number and takes the other's; bytes of another
// length under one number are ranks out of step. A peer that never comes
// fails the exchange after a minute.
struct Link {
  std::mutex              mu;
  std::condition_variable cv;
  struct Entry {
    std::vector<std::uint8_t> part[2];
    bool has[2] = {false, false};
    int  taken = 0;
  };
  std::map<std::uint64_t, Entry> at;
  int mismatched = 0;
};

struct PairChannel final : vpipe::genai::TpChannel {
  std::shared_ptr<Link> link;
  int me = 0;
  std::uint64_t exchanges = 0;
  PairChannel(std::shared_ptr<Link> l, int r) : link(std::move(l)), me(r) {}
  bool exchange(const void* partial, void* others, std::size_t bytes,
                vpipe::genai::TpDType, std::uint64_t seq,
                std::string* err) override
  {
    const auto* p = static_cast<const std::uint8_t*>(partial);
    std::unique_lock<std::mutex> lk(link->mu);
    Link::Entry& e = link->at[seq];
    e.part[me].assign(p, p + bytes);
    e.has[me] = true;
    link->cv.notify_all();
    if (!link->cv.wait_for(lk, std::chrono::seconds(60),
                           [&]() { return e.has[1 - me]; })) {
      if (err != nullptr) { *err = "the peer never came"; }
      return false;
    }
    if (e.part[1 - me].size() != bytes) {
      ++link->mismatched;
      if (err != nullptr) { *err = "the ranks are out of step"; }
      return false;
    }
    std::memcpy(others, e.part[1 - me].data(), bytes);
    ++exchanges;
    if (++e.taken == 2) { link->at.erase(seq); }
    return true;
  }
};

}  // namespace

TEST(hyperflow_h3_flash, runs_as_the_plain_h3_it_is_made_of)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Env noqmm("VPIPE_H3_NO_QMM_AUTOTUNE", "1");
  Env notile("VPIPE_H3_NO_ATTN_TILE_AUTOTUNE", "1");

  // hidden 256, 2 heads of 128 (inner 256), ffn 512, text 256; the time
  // MLP 256 -> 64 -> 16 (Flash's AdaLN reads 16); four release blocks.
  const int H = 256, A = 256, F = 512, TXT = 256, FREQ = 256, TH = 64,
            M = 16, REL = 4, DROP = 2, D = 18 * H, FD = 2 * H, RANK = 8;
  std::mt19937 rng(23);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto rnd = [&](std::size_t n, float sd, float mean = 0.0f) {
    std::vector<float> v(n);
    for (float& x : v) { x = mean + sd * nd(rng); }
    return v;
  };

  // ---- the model, by RELEASE block index ------------------------------
  std::vector<T> trunk;            // both files, names bare
  {
    std::vector<float> v;
    trunk.push_back(
        f32_("video_patch_proj.weight", {H, 96}, rnd(H * 96, 0.05f)));
    trunk.push_back(f32_("video_patch_proj.bias", {H}, rnd(H, 0.01f)));
    trunk.push_back(
        f32_("audio_patch_proj.weight", {H, 32}, rnd(H * 32, 0.05f)));
    trunk.push_back(f32_("audio_patch_proj.bias", {H}, rnd(H, 0.01f)));
    v = rnd((std::size_t)H * TXT, 0.03f);
    trunk.push_back(bf16_("condition_proj.weight", {H, TXT}, v));
    v = rnd(H, 0.01f);
    trunk.push_back(bf16_("condition_proj.bias", {H}, v));
    std::vector<float> inv(16);
    for (int i = 0; i < 16; ++i) {
      inv[(std::size_t)i] = 1.0f / std::pow(10000.0f, (float)(2 * i) / 32.0f);
    }
    trunk.push_back(f32_("rope.inv_freq", {16}, inv));
    const std::string b = "token_refiner.blocks.0.";
    v = rnd(H, 0.02f, 1.0f); trunk.push_back(bf16_(b + "norm1.weight", {H}, v));
    v = rnd(H, 0.02f, 1.0f); trunk.push_back(bf16_(b + "norm2.weight", {H}, v));
    v = rnd(128, 0.02f, 1.0f);
    trunk.push_back(bf16_(b + "attn.q_norm.weight", {128}, v));
    v = rnd(128, 0.02f, 1.0f);
    trunk.push_back(bf16_(b + "attn.k_norm.weight", {128}, v));
    v = rnd((std::size_t)3 * A * H, 0.04f);
    trunk.push_back(bf16_(b + "attn.qkv_proj.weight", {3 * A, H}, v));
    v = rnd((std::size_t)H * A, 0.04f);
    trunk.push_back(bf16_(b + "attn.out_proj.weight", {H, A}, v));
    v = rnd((std::size_t)2 * F * H, 0.04f);
    trunk.push_back(bf16_(b + "mlp.fc1.weight", {2 * F, H}, v));
    v = rnd((std::size_t)H * F, 0.04f);
    trunk.push_back(bf16_(b + "mlp.fc2.weight", {H, F}, v));
    v = rnd(H, 0.02f, 1.0f);
    trunk.push_back(bf16_("token_refiner.final_norm.weight", {H}, v));
    v = rnd(H, 0.02f, 1.0f);
    trunk.push_back(bf16_("final_layer.norm.weight", {H}, v));
    trunk.push_back(f32_("final_layer.video_out.weight", {96, H},
                         rnd(96 * H, 0.05f)));
    trunk.push_back(f32_("final_layer.video_out.bias", {96},
                         rnd(96, 0.01f)));
    trunk.push_back(f32_("final_layer.audio_out.weight", {32, H},
                         rnd(32 * H, 0.05f)));
    trunk.push_back(f32_("final_layer.audio_out.bias", {32},
                         rnd(32, 0.01f)));
    v = rnd((std::size_t)FD * M, 0.08f);
    trunk.push_back(bf16_("final_layer.adaln_proj.linear.weight", {FD, M}, v));
    v = rnd(FD, 0.05f);
    trunk.push_back(bf16_("final_layer.adaln_proj.linear.bias", {FD}, v));
  }
  // A low-rank pair: down [r, k], up [n, r] (bf16 both), and the dense
  // weight they are, up @ down, for R3.
  struct Pair { std::vector<float> down, up; int r; };
  // Factors scaled by the rank, so a wide pair's product is as large as
  // a narrow one's.
  auto lowrank = [&](int n, int k, int r) {
    const float sd = 0.2f * std::pow(16.0f / (float)r, 0.25f);
    Pair p{rnd((std::size_t)r * k, sd), rnd((std::size_t)n * r, sd), r};
    T td = bf16_("d", {r, k}, p.down);
    T tu = bf16_("u", {n, r}, p.up);
    (void)td; (void)tu;            // rounded in place; written below
    return p;
  };
  auto dense = [&](const Pair& p, int n, int k) {
    std::vector<float> w((std::size_t)n * k);
    for (int o = 0; o < n; ++o) {
      for (int i = 0; i < k; ++i) {
        double a = 0.0;
        for (int j = 0; j < p.r; ++j) {
          a += (double)p.up[(std::size_t)o * p.r + j] *
               p.down[(std::size_t)j * k + i];
        }
        w[(std::size_t)o * k + i] = (float)a;
      }
    }
    return w;
  };
  // Per release block: its tensors as R3 holds them (dense) and as F
  // holds them (low-rank where chosen), under the kept index.
  std::vector<T> r3, fl;
  for (int S = 0; S < REL; ++S) {
    if (S == DROP) {
      // Its weights exist in the release; neither file carries them.
      continue;
    }
    const int L = S < DROP ? S : S - 1;
    const std::string b = "blocks." + std::to_string(L) + ".";
    std::vector<float> v;
    auto both = [&](T t) {
      fl.push_back(t);
      r3.push_back(std::move(t));
    };
    v = rnd(H, 0.02f, 1.0f); both(bf16_(b + "norm1.weight", {H}, v));
    v = rnd(H, 0.02f, 1.0f); both(bf16_(b + "norm2.weight", {H}, v));
    v = rnd(128, 0.02f, 1.0f); both(bf16_(b + "attn.q_norm.weight", {128}, v));
    v = rnd(128, 0.02f, 1.0f); both(bf16_(b + "attn.k_norm.weight", {128}, v));
    v = rnd((std::size_t)D * M, 0.08f);
    both(bf16_(b + "adaln_proj.linear.weight", {D, M}, v));
    v = rnd(D, 0.05f);
    both(bf16_(b + "adaln_proj.linear.bias", {D}, v));
    // Block 0 all low-rank: a streamed run's slot 0 is BUILT from it and
    // then refilled with kept block 2's weights (refill_block_).
    const bool lr_all = S == 0;          // every projection low-rank
    // Block 1's q|k|v alone low-rank at a widest part rank of 192, under
    // a group (inner width 256) a multiple of 256: the matrix cores'
    // 256-wide banded twin.
    const bool lr_qkv = S == 1;
    const bool lr_fc2 = S == 3;          // fc2 alone
    auto proj = [&](const std::string& nm, int n, int k, bool low, int r) {
      if (!low) {
        v = rnd((std::size_t)n * k, 0.04f);
        both(bf16_(b + nm + ".weight", {n, k}, v));
        return;
      }
      Pair p = lowrank(n, k, r);
      fl.push_back(bf16_(b + nm + ".down.weight", {r, k}, p.down));
      fl.push_back(bf16_(b + nm + ".up.weight", {n, r}, p.up));
      std::vector<float> w = dense(p, n, k);
      r3.push_back(bf16_(b + nm + ".weight", {n, k}, w));
    };
    if (lr_all) {
      // q / k / v each its own pair, of different ranks (the banded
      // q|k|v pads them to the widest, 256: on an M5 the matrix cores'
      // 128-wide banded twin, as Flash's real part rank 1664 takes);
      // R3's fused q|k|v flat.
      Pair q = lowrank(A, H, 256), k = lowrank(A, H, 128),
           vv = lowrank(A, H, 160);
      fl.push_back(bf16_(b + "attn.q_proj.down.weight", {256, H}, q.down));
      fl.push_back(bf16_(b + "attn.q_proj.up.weight", {A, 256}, q.up));
      fl.push_back(bf16_(b + "attn.k_proj.down.weight", {128, H}, k.down));
      fl.push_back(bf16_(b + "attn.k_proj.up.weight", {A, 128}, k.up));
      fl.push_back(bf16_(b + "attn.v_proj.down.weight", {160, H}, vv.down));
      fl.push_back(bf16_(b + "attn.v_proj.up.weight", {A, 160}, vv.up));
      std::vector<float> w = dense(q, A, H);
      const std::vector<float> wk = dense(k, A, H), wv = dense(vv, A, H);
      w.insert(w.end(), wk.begin(), wk.end());
      w.insert(w.end(), wv.begin(), wv.end());
      r3.push_back(bf16_(b + "attn.qkv_proj.weight", {3 * A, H}, w));
    } else if (lr_qkv) {
      Pair q = lowrank(A, H, 192), k = lowrank(A, H, 96),
           vv = lowrank(A, H, 128);
      fl.push_back(bf16_(b + "attn.q_proj.down.weight", {192, H}, q.down));
      fl.push_back(bf16_(b + "attn.q_proj.up.weight", {A, 192}, q.up));
      fl.push_back(bf16_(b + "attn.k_proj.down.weight", {96, H}, k.down));
      fl.push_back(bf16_(b + "attn.k_proj.up.weight", {A, 96}, k.up));
      fl.push_back(bf16_(b + "attn.v_proj.down.weight", {128, H}, vv.down));
      fl.push_back(bf16_(b + "attn.v_proj.up.weight", {A, 128}, vv.up));
      std::vector<float> w = dense(q, A, H);
      const std::vector<float> wk = dense(k, A, H), wv = dense(vv, A, H);
      w.insert(w.end(), wk.begin(), wk.end());
      w.insert(w.end(), wv.begin(), wv.end());
      r3.push_back(bf16_(b + "attn.qkv_proj.weight", {3 * A, H}, w));
    } else {
      proj("attn.qkv_proj", 3 * A, H, false, 0);
    }
    // Plain pairs at 192 and 320: the matrix cores' 256-wide route
    // (128-255) and 128-wide one (256 and up), beside fc2's narrow one on
    // steel.
    proj("attn.out_proj", H, A, lr_all, 192);
    proj("mlp.fc1", 2 * F, H, lr_all, 320);
    proj("mlp.fc2", H, F, lr_all || lr_fc2, 24);
  }
  // The timestep MLPs: F's base and endpoint embedders, f32. R3 holds
  // the base one plus a flow-map adapter that makes the two: its base
  // factors move nothing (B = 0) and its endpoint factors are exactly
  // F's endpoint less the base -- rank FREQ / TH, so the difference is
  // carried whole.
  const std::vector<float> w1 = rnd((std::size_t)TH * FREQ, 0.15f);
  const std::vector<float> b1 = rnd(TH, 0.2f);
  const std::vector<float> w2 = rnd((std::size_t)M * TH, 0.3f);
  const std::vector<float> b2 = rnd(M, 0.2f);
  const std::vector<float> e1 = rnd((std::size_t)TH * FREQ, 0.15f);
  const std::vector<float> e2 = rnd((std::size_t)M * TH, 0.3f);
  for (std::vector<T>* ts : {&r3, &fl}) {
    ts->push_back(f32_("time_embedder.proj_in.weight", {TH, FREQ}, w1));
    ts->push_back(f32_("time_embedder.proj_in.bias", {TH}, b1));
    ts->push_back(f32_("time_embedder.proj_out.weight", {M, TH}, w2));
    ts->push_back(f32_("time_embedder.proj_out.bias", {M}, b2));
  }
  // The endpoint embedder keeps the base biases (a flow-map adapter
  // cannot move them).
  fl.push_back(f32_("endpoint_time_embedder.proj_in.weight", {TH, FREQ}, e1));
  fl.push_back(f32_("endpoint_time_embedder.proj_in.bias", {TH}, b1));
  fl.push_back(f32_("endpoint_time_embedder.proj_out.weight", {M, TH}, e2));
  fl.push_back(f32_("endpoint_time_embedder.proj_out.bias", {M}, b2));
  fl.push_back(f32_("hyperflow_gate", {1}, {0.25f}));
  for (const T& t : trunk) {
    r3.push_back(t);
    fl.push_back(t);
  }
  // R3 under the ComfyUI prefix: read as a DiT ComfyUI saved, qkv flat,
  // as F's bare names are.
  const std::string P = "model.diffusion_model.";
  for (T& t : r3) { t.name = P + t.name; }
  const fs::path fr = tmp_("r3.safetensors");
  const fs::path ff = tmp_("flash.safetensors");
  ASSERT_TRUE(write_st_(fr, r3));
  ASSERT_TRUE(write_st_(
      ff, fl,
      {{"hyperflow", "{\"format\":1,\"variant\":\"fl2va\",\"blocks\":3,"
                     "\"dropped_blocks\":[2]}"}}));
  // The flow map, f32 factors: A = I (rank = the input width), B = the
  // difference -- no alpha stated, so each pair applies as written. Its
  // header's gate F's.
  const fs::path lf = tmp_("flow.safetensors");
  {
    auto eye = [](int n) {
      std::vector<float> e((std::size_t)n * n, 0.0f);
      for (int i = 0; i < n; ++i) { e[(std::size_t)i * n + i] = 1.0f; }
      return e;
    };
    auto diff = [](const std::vector<float>& a, const std::vector<float>& b) {
      std::vector<float> d(a.size());
      for (std::size_t i = 0; i < a.size(); ++i) { d[i] = a[i] - b[i]; }
      return d;
    };
    std::vector<T> ts;
    ts.push_back(f32_("time_embedder.proj_in.lora_A.weight", {1, FREQ},
                      std::vector<float>(FREQ, 0.0f)));
    ts.push_back(f32_("time_embedder.proj_in.lora_B.weight", {TH, 1},
                      std::vector<float>(TH, 0.0f)));
    ts.push_back(f32_("time_embedder.proj_out.lora_A.weight", {1, TH},
                      std::vector<float>(TH, 0.0f)));
    ts.push_back(f32_("time_embedder.proj_out.lora_B.weight", {M, 1},
                      std::vector<float>(M, 0.0f)));
    ts.push_back(f32_("endpoint_time_embedder.proj_in.lora_A.weight",
                      {FREQ, FREQ}, eye(FREQ)));
    ts.push_back(f32_("endpoint_time_embedder.proj_in.lora_B.weight",
                      {TH, FREQ}, diff(e1, w1)));
    ts.push_back(f32_("endpoint_time_embedder.proj_out.lora_A.weight",
                      {TH, TH}, eye(TH)));
    ts.push_back(f32_("endpoint_time_embedder.proj_out.lora_B.weight",
                      {M, TH}, diff(e2, w2)));
    ASSERT_TRUE(write_st_(lf, ts,
                          {{"hyperflow", "true"},
                           {"hyperflow_gate", "0.25"},
                           {"hyperflow_sigmas", "[1.0, 0.5, 0.0]"}}));
  }
  // An ordinary adapter on the out projection, by RELEASE index: block 3
  // (F's 2) and block 2 (left out); its twin for R3 names block 3 as 2.
  const fs::path lrel = tmp_("lora-release.safetensors");
  const fs::path lkept = tmp_("lora-kept.safetensors");
  // The same draws for block 3's factors in both: block 2's (skipped on
  // F) come FIRST in the release file, so draw them apart.
  {
    std::mt19937 s2(91), s3(92);
    std::vector<T> ts;
    std::normal_distribution<float> n2(0.0f, 1.0f);
    auto put = [&](std::vector<T>* out, int name, std::mt19937& s) {
      std::vector<float> a((std::size_t)RANK * A), bb((std::size_t)H * RANK);
      for (float& x : a) { x = 1.5f * n2(s); }
      for (float& x : bb) { x = 1.5f * n2(s); }
      const std::string m =
          "blocks." + std::to_string(name) + ".attn.out_proj";
      out->push_back(bf16_(m + ".lora_A.weight", {RANK, A}, a));
      out->push_back(bf16_(m + ".lora_B.weight", {H, RANK}, bb));
    };
    std::mt19937 s3b = s3;
    put(&ts, 2, s2);
    put(&ts, 3, s3);
    ASSERT_TRUE(write_st_(lrel, ts));
    std::vector<T> tk;
    put(&tk, 2, s3b);              // release 3 is R3's kept 2
    ASSERT_TRUE(write_st_(lkept, tk));
  }

  // ---- runs ------------------------------------------------------------
  H3::Config rc, fc;
  std::string err;
  ASSERT_TRUE(H3::config_from_json(fr.string(), rc, &err));
  ASSERT_TRUE(H3::config_from_json(ff.string(), fc, &err));
  std::printf("[hyperflow_h3_flash] config: flash %d, %d blocks, dropped "
              "[%s], qkv %s, time dim %d\n", (int)fc.flash, fc.n_layers,
              fc.dropped_blocks.empty() ? ""
                                        : std::to_string(fc.dropped_blocks[0])
                                              .c_str(),
              fc.qkv_per_head ? "per head" : "flat", fc.time_dim);
  EXPECT_TRUE(fc.flash && !rc.flash && fc.n_layers == 3 &&
              rc.n_layers == 3 && fc.dropped_blocks == std::vector<int>{2} &&
              !fc.qkv_per_head && !rc.qkv_per_head && fc.time_dim == M &&
              fc.src_layer(2) == 3 && fc.kept_layer(2) == -1 &&
              fc.kept_layer(3) == 2);
  // 144 rows: past the matrix cores' floor (64), so an M5 runs the
  // pairs' GEMMs on them as a real clip does.
  const int latf = 2, lath = 16, latw = 16, naud = 8, ntext = 8;
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
  std::vector<float> uniq, ends;
  std::vector<int> ridx;
  h3::build_row_timesteps(lay, 0.6f, 0.5f, 1.0f, &uniq, &ridx);
  for (float t : uniq) { ends.push_back(std::min(1.0f, t + 0.2f)); }
  struct Out {
    std::vector<std::uint16_t> v;
    bool loaded = false;
  };
  auto run = [&](const fs::path& file, const H3::Config& cfg,
                 const std::vector<H3::LoraSpec>& loras, bool stream,
                 const std::vector<float>* endpoints) -> Out {
    Out o;
    std::unique_ptr<H3> m;
    try {
      m = H3::load(file.string(), mc, cfg, stream, loras);
    } catch (const std::exception& e) {
      std::printf("[hyperflow_h3_flash] refused: %s\n", e.what());
    }
    if (m == nullptr) { return o; }
    o.loaded = true;
    // Pure streaming: no block promoted, so every forward refills.
    if (stream) { m->set_residency_reserve(0); }
    m->set_gemm_route(H3::GemmRoute::kSteelBm32);
    H3::Velocity out;
    for (int i = 0; i < 2; ++i) {   // the second compared
      H3::Step st;
      st.video = &vb;
      st.audio = &abuf;
      st.text = &tb;
      st.layout = &lay;
      st.timesteps = &uniq;
      st.row_timestep_index = &ridx;
      st.endpoints = endpoints;
      std::string ferr;
      out = m->forward(st, &ferr);
      if (out.empty()) {
        std::printf("[hyperflow_h3_flash] forward: %s\n", ferr.c_str());
        return o;
      }
    }
    const auto* p = static_cast<const std::uint16_t*>(out.video.contents());
    o.v.assign(p, p + out.video.byte_size() / 2);
    return o;
  };
  const Out ref = run(fr, rc, {{lf.string(), 1.0f}}, false, &ends);
  const Out plain = run(fr, rc, {}, false, nullptr);
  const Out f0 = run(ff, fc, {}, false, &ends);
  const Out fs0 = run(ff, fc, {}, true, &ends);
  // Without endpoints: Flash has no single-time mode.
  const Out fnone = run(ff, fc, {}, false, nullptr);
  ASSERT_TRUE(!ref.v.empty() && !f0.v.empty() && !plain.v.empty());
  const double base = rel_(f0.v, ref.v);
  const double two = rel_(ref.v, plain.v);
  std::printf("[hyperflow_h3_flash] F against R3 under its flow map: %.3g "
              "preloaded, %.3g streamed (the two-time embedding moves R3 "
              "%.3g); with no endpoints F %s\n", base, rel_(fs0.v, ref.v),
              two, fnone.v.empty() ? "refuses" : "RUNS");
  EXPECT_TRUE(two > 0.05);
  EXPECT_TRUE(base < 2e-2);
  EXPECT_TRUE(rel_(fs0.v, ref.v) < 2e-2);
  EXPECT_TRUE(fnone.v.empty());
  // The streamed forms are ONE function: the slot pair (slot 0 built
  // from low-rank block 0, refilled with block 2's weights), a single
  // slot, and the per-block allocations -- byte for byte.
  Out fs1, fsn;
  {
    Env one("VPIPE_H3_SINGLE_SLOT", "1");
    fs1 = run(ff, fc, {}, true, &ends);
  }
  {
    Env none("VPIPE_H3_NO_SLOTS", "1");
    fsn = run(ff, fc, {}, true, &ends);
  }
  const bool same1 = !fs0.v.empty() && fs1.v == fs0.v;
  const bool samen = !fs0.v.empty() && fsn.v == fs0.v;
  std::printf("[hyperflow_h3_flash] streamed: single slot %s, no slots %s "
              "the slot pair's bytes; preloaded %s them\n",
              same1 ? "=" : "!=", samen ? "=" : "!=",
              f0.v == fs0.v ? "=" : "!=");
  EXPECT_TRUE(same1);
  EXPECT_TRUE(samen);
  EXPECT_TRUE(f0.v == fs0.v);

  // An adapter by the release's block indices: on F its block-3 module
  // lands on kept block 2, its block-2 module is skipped; on R3 the twin
  // names block 2 directly.
  const Out fa = run(ff, fc, {{lrel.string(), 1.0f}}, false, &ends);
  const Out ra = run(fr, rc, {{lf.string(), 1.0f}, {lkept.string(), 1.0f}},
                     false, &ends);
  const double da = rel_delta_(fa.v, f0.v, ra.v, ref.v);
  std::printf("[hyperflow_h3_flash] an adapter by release index: its "
              "effect on F %.3g from its renumbered twin's on R3 (it moves "
              "R3 %.3g)\n", da, rel_(ra.v, ref.v));
  EXPECT_TRUE(rel_(ra.v, ref.v) > 5.0 * base && da < 5e-2);
  // A flow-map adapter on Flash, which blends two times of its own:
  // refused.
  const Out fflow = run(ff, fc, {{lf.string(), 1.0f}}, false, &ends);
  EXPECT_TRUE(!fflow.loaded);

  // ---- pi-H3 on Flash: the stock pruned curve (Config::curve_from) ---
  // A stock PRUNED DiT S holds a curve table and one AdaLN a RELEASE
  // block (all four); Flash under it must run as R3's blocks under S's
  // AdaLN by release index -- RP, the pruned DiT made of them -- single
  // time. Reading S's AdaLN by the KEPT index puts block 2's (the one
  // left out) on kept block 2: RPK, which this must not match.
  const int GRID = 1025;
  std::vector<float> table((std::size_t)GRID * 8);
  for (int g = 0; g < GRID; ++g) {
    for (int k = 0; k < 8; ++k) {
      table[(std::size_t)g * 8 + k] =
          std::sin(0.004f * (float)g * (float)(k + 1) + 0.7f * (float)k);
    }
  }
  std::vector<std::vector<float>> aw, ab;   // by release index
  for (int S = 0; S < REL; ++S) {
    // The block left out's AdaLN far from the rest: read in kept block
    // 2's place (by the kept index), it is plain in the output.
    const float k = S == DROP ? 4.0f : 1.0f;
    aw.push_back(rnd((std::size_t)D * 8, 0.3f * k));
    ab.push_back(rnd(D, 0.2f * k));
  }
  std::vector<float> faw = rnd((std::size_t)FD * 8, 0.08f);
  std::vector<float> fab = rnd(FD, 0.05f);
  std::vector<T> st;
  // Bare names, as the stock pruned file stores them.
  st.push_back(f32_("adaln_t_table", {GRID, 8}, table));
  for (int S = 0; S < REL; ++S) {
    const std::string b = "blocks." + std::to_string(S) +
                          ".adaln_proj.linear.";
    st.push_back(bf16_(b + "weight", {D, 8}, aw[(std::size_t)S]));
    st.push_back(bf16_(b + "bias", {D}, ab[(std::size_t)S]));
  }
  st.push_back(bf16_("final_layer.adaln_proj.linear.weight", {FD, 8},
                     faw));
  st.push_back(bf16_("final_layer.adaln_proj.linear.bias", {FD}, fab));
  const fs::path fst = tmp_("stock.safetensors");
  ASSERT_TRUE(write_st_(fst, st));
  auto pruned_of = [&](bool by_kept) {
    std::vector<T> rp;
    for (const T& t : r3) {
      if (t.name.find("adaln_proj") != std::string::npos ||
          t.name.find("time_embedder") != std::string::npos) {
        continue;
      }
      rp.push_back(t);
    }
    rp.push_back(f32_(P + "adaln_t_table", {GRID, 8}, table));
    for (int L = 0; L < fc.n_layers; ++L) {
      const int S = by_kept ? L : fc.src_layer(L);
      const std::string b = P + "blocks." + std::to_string(L) +
                            ".adaln_proj.linear.";
      std::vector<float> w = aw[(std::size_t)S], bb = ab[(std::size_t)S];
      rp.push_back(bf16_(b + "weight", {D, 8}, w));
      rp.push_back(bf16_(b + "bias", {D}, bb));
    }
    std::vector<float> w = faw, bb = fab;
    rp.push_back(bf16_(P + "final_layer.adaln_proj.linear.weight", {FD, 8},
                       w));
    rp.push_back(bf16_(P + "final_layer.adaln_proj.linear.bias", {FD}, bb));
    return rp;
  };
  const fs::path frp = tmp_("rp.safetensors");
  const fs::path frk = tmp_("rpk.safetensors");
  ASSERT_TRUE(write_st_(frp, pruned_of(false)));
  ASSERT_TRUE(write_st_(frk, pruned_of(true)));
  H3::Config pc, kc, cc = fc;
  ASSERT_TRUE(H3::config_from_json(frp.string(), pc, &err));
  ASSERT_TRUE(H3::config_from_json(frk.string(), kc, &err));
  cc.curve_from = fst.string();
  std::string cerr;
  const bool on_curve = H3::on_stock_curve(cc, &cerr);
  std::printf("[hyperflow_h3_flash] the stock curve: %s (grid %d, AdaLN "
              "input %d); RP pruned %d\n",
              on_curve ? "taken" : cerr.c_str(), cc.curve_grid, cc.time_dim,
              (int)pc.pruned());
  EXPECT_TRUE(on_curve && cc.pruned() && pc.pruned() &&
              cc.curve_grid == GRID && cc.time_dim == 8);
  const Out rp = run(frp, pc, {}, false, nullptr);
  const Out rk = run(frk, kc, {}, false, nullptr);
  const Out fcur = run(ff, cc, {}, false, nullptr);
  const Out fcurs = run(ff, cc, {}, true, nullptr);
  // A flow-map adapter on Flash under the stock curve: refused (single
  // time, nothing for it to blend against).
  // (The released timestep MLP named, so the pruned path's own check
  // passes and Flash's is what refuses.)
  H3::Config cm = cc;
  cm.time_mlp_dir = fr.string();
  const Out fcflow = run(ff, cm, {{lf.string(), 1.0f}}, false, &ends);
  ASSERT_TRUE(!rp.v.empty() && !rk.v.empty() && !fcur.v.empty());
  const double dc = rel_(fcur.v, rp.v), dk = rel_(rk.v, rp.v);
  std::printf("[hyperflow_h3_flash] Flash on the stock curve against RP: "
              "%.3g preloaded, %.3g streamed (the AdaLN by kept index moves "
              "RP %.3g); a flow map on it %s\n", dc,
              rel_(fcurs.v, rp.v), dk, fcflow.loaded ? "LOADS" : "refused");
  EXPECT_TRUE(dc < 2e-2);
  EXPECT_TRUE(rel_(fcurs.v, rp.v) < 2e-2);
  EXPECT_TRUE(dk > 5.0 * dc && dk > 0.05);
  EXPECT_TRUE(!fcflow.loaded);

  // ---- SPLIT: two ranks of F, a head and 256 feed-forward rows each ----
  // Each rank reads F's low-rank pairs whole and cuts them as an adapter's
  // factors are cut: q|k|v's and fc1's ups by the share's rows (q|k|v's
  // bands then a head wide), the out projection's and fc2's downs by its
  // columns -- their products partial sums the exchanges complete, as the
  // dense blocks' are. A rank holds no AdaLN projection, so each run here
  // is BAKED for the one step (F's two times; the stock curve's runs
  // unbaked on a rank as unsplit), the unsplit model run the same way.
  // The bar: two bf16 roundings of a partial sum where the unsplit model
  // rounds once -- and each rank's low-rank intermediate (down @ x over
  // its columns) rounded apart. MEASURED on the M5 Max: R3 split dense
  // 6.3e-3, F 9.0e-3 (two times) and 9.6e-3 (the stock curve). The two
  // ranks are IDENTICAL. MUTATIONS, each failing it: a rank keeping
  // q|k|v's whole band group 1.28-1.40; fc2's down left whole on a rank
  // 1.28-1.31.
  auto rank_cfg = [&](const H3::Config& cfg, int r) {
    H3::Config c = cfg;
    c.tp = vpipe::genai::TpShare{};
    c.tp.rank = r;
    c.tp.ranks = 2;
    c.tp.n_heads = cfg.n_heads;
    c.tp.n_ffn = cfg.ffn;
    c.tp.head0 = r == 0 ? 0 : 1;
    c.tp.head1 = r == 0 ? 1 : cfg.n_heads;
    c.tp.ffn0 = r == 0 ? 0 : cfg.ffn / 2;
    c.tp.ffn1 = r == 0 ? cfg.ffn / 2 : cfg.ffn;
    return c;
  };
  // One model loaded (baked where `bake`), its second forward's velocity.
  auto load_ = [&](const fs::path& file, const H3::Config& cfg, bool stream,
                   bool bake, const std::vector<float>* endpoints) {
    std::unique_ptr<H3> m;
    try {
      m = H3::load(file.string(), mc, cfg, stream, {});
    } catch (const std::exception& e) {
      std::printf("[hyperflow_h3_flash] refused: %s\n", e.what());
    }
    if (m == nullptr) { return m; }
    if (stream) { m->set_residency_reserve(0); }
    m->set_gemm_route(H3::GemmRoute::kSteelBm32);
    std::string berr;
    if (bake) {
      std::vector<std::vector<float>> eps;
      if (endpoints != nullptr) { eps.push_back(*endpoints); }
      if (!m->bake_adaln({uniq}, &berr,
                         endpoints != nullptr ? &eps : nullptr)) {
        std::printf("[hyperflow_h3_flash] bake: %s\n", berr.c_str());
        return std::unique_ptr<H3>{};
      }
    }
    return m;
  };
  auto fwd = [&](H3& m, bool bake, const std::vector<float>* endpoints,
                 std::vector<std::uint16_t>* v) {
    H3::Velocity out;
    for (int i = 0; i < 2; ++i) {
      H3::Step st;
      st.video = &vb;
      st.audio = &abuf;
      st.text = &tb;
      st.layout = &lay;
      st.timesteps = &uniq;
      st.row_timestep_index = &ridx;
      st.endpoints = endpoints;
      if (bake) { st.schedule_index = 0; }
      std::string ferr;
      out = m.forward(st, &ferr);
      if (out.empty()) {
        std::printf("[hyperflow_h3_flash] forward: %s\n", ferr.c_str());
        return false;
      }
    }
    const auto* p = static_cast<const std::uint16_t*>(out.video.contents());
    v->assign(p, p + out.video.byte_size() / 2);
    return true;
  };
  struct Split {
    std::vector<std::uint16_t> whole, v[2];
    std::size_t held[2] = {}, booked = 0;
    // A streamed rank's per-clip holding (generate-video's correction
    // adds it to the set's) and the pairs it holds as its own buffers.
    std::size_t holding = 0, pairs = 0;
    int mismatched = 0;
    std::uint64_t exchanges = 0;
  };
  auto split = [&](const fs::path& file, const H3::Config& cfg,
                   bool stream, bool bake,
                   const std::vector<float>* endpoints, Split* s) {
    {
      auto m = load_(file, cfg, stream, bake, endpoints);
      if (m == nullptr || !fwd(*m, bake, endpoints, &s->whole)) {
        return false;
      }
    }
    auto link = std::make_shared<Link>();
    std::shared_ptr<PairChannel> ch[2];
    std::unique_ptr<H3> m[2];
    for (int r = 0; r < 2; ++r) {
      m[r] = load_(file, rank_cfg(cfg, r), stream, bake, endpoints);
      if (m[r] == nullptr) { return false; }
      ch[r] = std::make_shared<PairChannel>(link, r);
      std::string berr;
      if (!m[r]->begin_tp_generation(ch[r], &berr)) {
        std::printf("[hyperflow_h3_flash] begin: %s\n", berr.c_str());
        return false;
      }
    }
    bool ok1 = false;
    std::thread t([&]() { ok1 = fwd(*m[1], bake, endpoints, &s->v[1]); });
    const bool ok0 = fwd(*m[0], bake, endpoints, &s->v[0]);
    t.join();
    for (int r = 0; r < 2; ++r) { s->held[r] = m[r]->held_bytes(); }
    s->holding = m[0]->streamed_holding_bytes();
    s->pairs = m[0]->lowrank_own_bytes();
    // What the plan books for rank 0 (its pairs at their cut sizes,
    // resident), against what it holds.
    s->booked = H3::tp_held_bytes(file.string(), rank_cfg(cfg, 0), 0, mc);
    s->mismatched = link->mismatched;
    s->exchanges = ch[0]->exchanges;
    return ok0 && ok1;
  };
  auto arm = [&](const char* name, const fs::path& file,
                 const H3::Config& cfg, bool stream, bool bake,
                 const std::vector<float>* endpoints) {
    Split s;
    ASSERT_TRUE(split(file, cfg, stream, bake, endpoints, &s));
    const double d = rel_(s.v[0], s.whole);
    std::printf("[hyperflow_h3_flash] split, %-24s against unsplit %.3g, "
                "ranks %s, %llu exchanges, rank 0 holds %zu KB (booked "
                "%zu)\n", name, d, s.v[0] == s.v[1] ? "IDENTICAL" : "DIFFER",
                (unsigned long long)s.exchanges, s.held[0] >> 10,
                s.booked >> 10);
    EXPECT_TRUE(s.mismatched == 0 && s.exchanges > 0);
    EXPECT_TRUE(!s.v[0].empty() && s.v[0] == s.v[1]);
    EXPECT_TRUE(d < 1e-2);
    // (MUTATION: the pairs booked whole on a rank: 4,560 KB.)
    EXPECT_TRUE(s.booked > 0 &&
                std::fabs((double)s.booked - (double)s.held[0]) <
                    0.1 * (double)s.held[0]);
    // A rank's pairs are its own buffers, never the set's, so a streamed
    // rank's holding beside the set (what generate-video adds to the
    // set's after each clip) carries them: what the plan books past it --
    // the set's trunk and the CPU's f32 -- is less than they are.
    // (MUTATION: the pairs counted in held_bytes() alone, as before:
    // 1,959 KB booked past the holding against 1,680 of pairs.)
    if (stream) {
      std::printf("[hyperflow_h3_flash]   streamed rank 0: holding %zu KB, "
                  "its pairs %zu KB\n", s.holding >> 10, s.pairs >> 10);
    }
    if (stream && s.pairs > 0) {
      EXPECT_TRUE(s.holding >= s.pairs && s.booked > s.holding &&
                  s.booked - s.holding < s.pairs);
    }
  };
  // The bar's measure: R3, its pairs multiplied out, split as any dense
  // DiT is -- what two roundings cost on a model this small.
  arm("dense R3 (the rounding)", fr, rc, false, true, nullptr);
  arm("two times, preloaded", ff, fc, false, true, &ends);
  arm("two times, streamed", ff, fc, true, true, &ends);
  arm("the stock curve", ff, cc, false, false, nullptr);
  arm("the stock curve, streamed", ff, cc, true, false, nullptr);

  for (const fs::path& f : {fr, ff, lf, lrel, lkept, fst, frp, frk}) {
    fs::remove(f);
  }
}

// FLASH SPLIT ON THE REAL FILE: the release's FL2VA single file -- int8
// ConvRot projections read as stored on the Per Token tier, 68 low-rank
// pairs cut to each rank's share, three blocks left out -- two ranks of
// 28 heads and 7,168 feed-forward rows each over an in-process link,
// streamed as generate-video runs it, baked for the one step (a rank
// holds no AdaLN projection). Against the unsplit model the same way: the
// ranks IDENTICAL, and the split within twice what the tier itself does
// to the unsplit model (its Per Token forward against the codes widened
// per GEMM) -- a rank quantizes its activations over its own columns
// where the unsplit model quantizes over all of them, as the pi-H3 split
// over the pruned int8 DiT is held. And the plan books a rank at about
// half the file, its pairs' downs (q|k|v, fc1) whole.
// Env: VPIPE_H3_FLASH_DIT (minimax_h3_fl2va_hyperflow_flash_14b-int8_
// convrot.safetensors).
TEST(hyperflow_h3_flash, splits_on_the_real_file)
{
  const char* dit = std::getenv("VPIPE_H3_FLASH_DIT");
  if (dit == nullptr) {
    std::printf("  SKIP: VPIPE_H3_FLASH_DIT unset\n");
    return;
  }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Env noqmm("VPIPE_H3_NO_QMM_AUTOTUNE", "1");
  Env notile("VPIPE_H3_NO_ATTN_TILE_AUTOTUNE", "1");
  Env minm("VPIPE_I8_GEMM_MIN_M", "1");   // a clip's native path
  H3::Config fc;
  std::string err;
  ASSERT_TRUE(H3::config_from_json(dit, fc, &err));
  ASSERT_TRUE(fc.flash);
  fc.i8_gemm = true;
  fc.i8_tier = 2;   // token: ConvRot's codes as stored
  fc.autotune = false;
  const int latf = 2, lath = 8, latw = 8, naud = 8, ntext = 8;
  h3::PackedLayout lay;
  const std::vector<int> tags((std::size_t)ntext, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(tags, latf, lath, latw, naud,
                                        fc.patch_h, fc.patch_w,
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
      lay.video_indices.size() * (std::size_t)fc.video_patch_elems(), 0.017f);
  const metal_compute::SharedBuffer abuf = input(
      (std::size_t)lay.num_audio_rows * (std::size_t)fc.audio_channels,
      0.031f);
  const metal_compute::SharedBuffer tb =
      input((std::size_t)ntext * fc.text_dim, 0.005f);
  std::vector<float> uniq, ends;
  std::vector<int> ridx;
  h3::build_row_timesteps(lay, 0.6f, 0.45f, 0.999f, &uniq, &ridx);
  for (float t : uniq) { ends.push_back(std::min(1.0f, t + 0.25f)); }
  auto load = [&](const H3::Config& c, H3::GemmRoute route) {
    std::unique_ptr<H3> m = H3::load(dit, mc, c, /*stream=*/true, {});
    if (m == nullptr) { return m; }
    m->set_residency_reserve(0);
    m->set_gemm_route(route);
    std::string berr;
    std::vector<std::vector<float>> eps = {ends};
    if (!m->bake_adaln({uniq}, &berr, &eps)) {
      std::printf("  bake: %s\n", berr.c_str());
      return std::unique_ptr<H3>{};
    }
    return m;
  };
  auto fwd = [&](H3& m) {
    H3::Step st;
    st.video = &vb;
    st.audio = &abuf;
    st.text = &tb;
    st.layout = &lay;
    st.timesteps = &uniq;
    st.row_timestep_index = &ridx;
    st.endpoints = &ends;
    st.schedule_index = 0;
    std::string ferr;
    H3::Velocity v = m.forward(st, &ferr);
    if (v.empty()) {
      std::printf("  forward: %s\n", ferr.c_str());
      return std::vector<std::uint16_t>{};
    }
    const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
    return std::vector<std::uint16_t>(p, p + v.video.byte_size() / 2);
  };
  std::vector<std::uint16_t> whole, widened;
  {
    auto m = load(fc, H3::GemmRoute::kSteelBm32);
    ASSERT_TRUE(m != nullptr);
    whole = fwd(*m);
  }
  {
    H3::Config cw = fc;
    cw.i8_gemm = false;   // ConvRot widened per GEMM: no int8 tier
    cw.i8_tier = 0;
    auto m = load(cw, H3::GemmRoute::kSteelBm32);
    ASSERT_TRUE(m != nullptr);
    widened = fwd(*m);
  }
  const int h = fc.n_heads / 2, f = fc.ffn / 2;
  auto link = std::make_shared<Link>();
  std::shared_ptr<PairChannel> ch[2];
  std::unique_ptr<H3> m[2];
  std::size_t booked = 0;
  for (int r = 0; r < 2; ++r) {
    H3::Config c = fc;
    c.tp = vpipe::genai::TpShare{};
    c.tp.rank = r;
    c.tp.ranks = 2;
    c.tp.n_heads = fc.n_heads;
    c.tp.n_ffn = fc.ffn;
    c.tp.head0 = r == 0 ? 0 : h;
    c.tp.head1 = r == 0 ? h : fc.n_heads;
    c.tp.ffn0 = r == 0 ? 0 : f;
    c.tp.ffn1 = r == 0 ? f : fc.ffn;
    if (r == 0) { booked = H3::tp_held_bytes(dit, c, 0, mc); }
    m[r] = load(c, H3::GemmRoute::kSteelBm32);
    ASSERT_TRUE(m[r] != nullptr);
    ch[r] = std::make_shared<PairChannel>(link, r);
    std::string berr;
    ASSERT_TRUE(m[r]->begin_tp_generation(ch[r], &berr));
  }
  std::vector<std::uint16_t> v[2];
  std::thread t1([&]() { v[1] = fwd(*m[1]); });
  v[0] = fwd(*m[0]);
  t1.join();
  ASSERT_TRUE(!whole.empty() && !widened.empty() && !v[0].empty() &&
              !v[1].empty());
  const double d = rel_(v[0], whole), own = rel_(whole, widened);
  const std::size_t file = (std::size_t)fs::file_size(dit);
  std::printf("[hyperflow_h3_flash] the real file split %d / %d heads, %d / "
              "%d ffn rows, streamed: %.3g from the unsplit model (the tier "
              "itself moves it %.3g from the widened codes); the ranks %s; "
              "%llu exchanges; rank 0 booked %zu MB of the file's %zu\n", h,
              fc.n_heads - h, f, fc.ffn - f, d, own,
              v[0] == v[1] ? "IDENTICAL" : "DIFFER",
              (unsigned long long)ch[0]->exchanges, booked >> 20,
              file >> 20);
  EXPECT_TRUE(v[0] == v[1]);
  EXPECT_TRUE(link->mismatched == 0 && ch[0]->exchanges > 0);
  EXPECT_TRUE(d < std::max(1e-2, 2.0 * own));
  EXPECT_TRUE(booked > file / 3 && booked < file * 2 / 3);
}
