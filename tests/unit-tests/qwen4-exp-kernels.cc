// Qwen3.8-Flash-Next (model_type qwen4_exp) tests.
//
// qwen4_exp_kernels.*: no checkpoint -- the 5-bit routed experts (MLX
// affine w5 g64: 8 codes in 5 bytes, a little-endian bit stream), every
// route a 5-bit expert takes, against an f64 reference over random codes.
// GEOMETRY is the model's own expert: hidden 2560 into an inner of 640, so
// the gate|up GEMVs run whole 512-wide blocks and the down GEMV (K = 640)
// ends in a PARTIAL block, whose idle lanes must read nothing past the row.
//
// qwen4_exp.*: the checkpoint against the reference's own forward
// (tools/dump_qwen4_exp_golden.py: transformers' Qwen4ExpTextModel, a
// layer at a time, the checkpoint's affine codes dequantized exactly; fp32
// and bf16 -- the bf16 run's distance from fp32 is the bar).
//   VPIPE_QWEN4_EXP_TEST_MODEL_PATH  the checkpoint directory
//   VPIPE_QWEN4_EXP_GOLDEN           the golden directory
//   VPIPE_QWEN4_EXP_CACHE_MB         the expert cache (default 36864; 0 =
//                                    every expert resident, ~90 GB)
//   VPIPE_QWEN4_EXP_F16              run f16 instead of bf16
//   VPIPE_QWEN4_EXP_DFLASH           a DFlash 2 drafter for it
//                                    (incoai/Qwen3.8-Flash-Next-DFlash2)

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"
#include "generative-models/model-loader.h"
#include "generative-models/moe-expert-cache.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/qwen3/metal-qwen-vision.h"
#include "generative-models/chat-template.h"
#include "generative-models/shared/dflash-drafter.h"
#include "generative-models/tokenizer.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;
using metal_compute::MetalCompute;

namespace {

std::uint16_t
to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t r = ((u >> 16) & 1u) + 0x7FFFu;
  return (std::uint16_t)((u + r) >> 16);
}

double
from_bf16_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::uint16_t
to_f16_(double d)
{
  const _Float16 h = (_Float16)d;
  std::uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}

double
from_f16_(std::uint16_t u)
{
  _Float16 h;
  std::memcpy(&h, &u, 2);
  return (double)h;
}

// MLX's affine 5-bit row: each run of 8 codes is 40 bits, code j at bit
// 5j, little-endian -- 5 bytes. K must be a multiple of 8.
void
pack5_row_(const std::uint8_t* q, int K, std::uint8_t* out)
{
  for (int p = 0; p < K / 8; ++p) {
    std::uint64_t v = 0;
    for (int j = 0; j < 8; ++j) {
      v |= (std::uint64_t)(q[p * 8 + j] & 31u) << (5 * j);
    }
    for (int b = 0; b < 5; ++b) {
      out[p * 5 + b] = (std::uint8_t)(v >> (8 * b));
    }
  }
}

}  // namespace

// The routed experts at 5 bits, every route: the per-pair gathers (decode,
// and prefill below the grouped crossover), the grouped GEMV (MAXM 2 sorted
// rows a weight read) and the steel grouped GEMMs (32-row tiles), at f16
// and at bf16. Gate|up are interleaved by feature (row 2g gate g, 2g+1 up
// g), as the loader lays a layer's slab. Several experts with ragged pair
// counts, so pad slots and a pad tile are exercised.
TEST(qwen4_exp_kernels, w5_experts_match_f64)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  using metal_compute::CommandStream;
  using metal_compute::ComputeEncoder;
  using metal_compute::SharedBuffer;
  constexpr int E = 6, H = 2560, I = 640, M = 75, TOPK = 4, G = 64;
  constexpr int NP = M * TOPK;
  std::uint32_t rng = 20261020u;
  auto next = [&]() {
    rng = rng * 1664525u + 1013904223u;
    return rng;
  };
  auto unif = [&]() { return (next() >> 8) * (1.0 / 16777216.0); };
  // Codes, and per-group scales / biases as bf16 values (exact in f16 too
  // at these magnitudes, so both element types see the same weights).
  std::vector<std::uint8_t> qgu((std::size_t)E * 2 * I * H),
      qdn((std::size_t)E * H * I);
  for (auto& c : qgu) { c = (std::uint8_t)(next() >> 27); }
  for (auto& c : qdn) { c = (std::uint8_t)(next() >> 27); }
  auto mk_sb = [&](std::size_t n, std::vector<std::uint16_t>& s,
                   std::vector<std::uint16_t>& b) {
    s.resize(n);
    b.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      const float sv = (float)(2e-3 + 4e-3 * unif());
      s[i] = to_bf16_(sv);
      b[i] = to_bf16_((float)(-16.0 * sv * (0.8 + 0.4 * unif())));
    }
  };
  std::vector<std::uint16_t> sgu, bgu, sdn, bdn;
  mk_sb((std::size_t)E * 2 * I * (H / G), sgu, bgu);
  mk_sb((std::size_t)E * H * (I / G), sdn, bdn);
  std::vector<std::uint16_t> xb((std::size_t)M * H);
  for (auto& v : xb) { v = to_bf16_((float)(2.0 * unif() - 1.0)); }
  std::vector<std::int32_t> eid(NP);
  for (int p = 0; p < NP; ++p) {
    const int r = (int)(next() % 23u);
    eid[p] = r < 20 ? r % 5 : 5;      // expert 5 rare: mostly padding
  }
  // Packed weights, MLX layout: [E, rows, K*5/8] bytes.
  std::vector<std::uint8_t> wgu((std::size_t)E * 2 * I * H * 5 / 8),
      wdn((std::size_t)E * H * I * 5 / 8);
  for (std::size_t r = 0; r < (std::size_t)E * 2 * I; ++r) {
    pack5_row_(&qgu[r * H], H, &wgu[r * H * 5 / 8]);
  }
  for (std::size_t r = 0; r < (std::size_t)E * H; ++r) {
    pack5_row_(&qdn[r * I], I, &wdn[r * I * 5 / 8]);
  }
  auto deq = [](const std::vector<std::uint8_t>& q,
                const std::vector<std::uint16_t>& s,
                const std::vector<std::uint16_t>& b, std::size_t row, int K,
                int k) {
    const std::size_t g = row * (K / G) + (std::size_t)(k / G);
    return from_bf16_(s[g]) * (double)q[row * K + k] + from_bf16_(b[g]);
  };
  // f64 gate|up -> silu(g)*u, unrounded; the down reference rounds act to
  // each run's element type, so the down kernels answer for their own math.
  std::vector<double> act_ref((std::size_t)NP * I);
  for (int p = 0; p < NP; ++p) {
    const int e = eid[p], t = p / TOPK;
    for (int g = 0; g < I; ++g) {
      double a = 0.0, u = 0.0;
      const std::size_t rg = (std::size_t)e * 2 * I + 2 * g;
      for (int k = 0; k < H; ++k) {
        const double xv = from_bf16_(xb[(std::size_t)t * H + k]);
        a += xv * deq(qgu, sgu, bgu, rg, H, k);
        u += xv * deq(qgu, sgu, bgu, rg + 1, H, k);
      }
      act_ref[(std::size_t)p * I + g] = a / (1.0 + std::exp(-a)) * u;
    }
  }
  // Sorted slots: per expert, its pairs in pair order padded to whole tiles
  // of `tile` rows, then one pad tile.
  struct Sorted {
    std::vector<std::int32_t> srow, sdst, t2e;
  };
  auto sort_ = [&](int tile) {
    Sorted s;
    for (int e = 0; e < E; ++e) {
      std::vector<int> ps;
      for (int p = 0; p < NP; ++p) {
        if (eid[p] == e) { ps.push_back(p); }
      }
      const int tiles = ((int)ps.size() + tile - 1) / tile;
      for (int t = 0; t < tiles; ++t) { s.t2e.push_back(e); }
      for (int k = 0; k < tiles * tile; ++k) {
        s.srow.push_back(k < (int)ps.size() ? ps[k] / TOPK : -1);
        s.sdst.push_back(k < (int)ps.size() ? ps[k] : -1);
      }
    }
    s.t2e.push_back(-1);
    for (int k = 0; k < tile; ++k) {
      s.srow.push_back(-1);
      s.sdst.push_back(-1);
    }
    return s;
  };
  const Sorted s2 = sort_(2), s32 = sort_(32);
  auto upload = [&](const void* p, std::size_t n) {
    SharedBuffer b = mc->make_shared_buffer(n);
    std::memcpy(b.contents(), p, n);
    return b;
  };
  SharedBuffer bwgu = upload(wgu.data(), wgu.size()),
               bwdn = upload(wdn.data(), wdn.size()),
               beid = upload(eid.data(), eid.size() * 4);
  for (const bool bf16 : {false, true}) {
    // The element type's view of the scales, biases and input.
    auto conv = [&](const std::vector<std::uint16_t>& v) {
      std::vector<std::uint16_t> o(v.size());
      for (std::size_t i = 0; i < v.size(); ++i) {
        o[i] = bf16 ? v[i] : to_f16_(from_bf16_(v[i]));
      }
      return o;
    };
    auto rd = [&](std::uint16_t u) {
      return bf16 ? from_bf16_(u) : from_f16_(u);
    };
    auto rnd = [&](double d) {
      return bf16 ? from_bf16_(to_bf16_((float)d)) : from_f16_(to_f16_(d));
    };
    std::vector<double> part_ref((std::size_t)NP * H);
    for (int p = 0; p < NP; ++p) {
      const std::size_t r0 = (std::size_t)eid[p] * H;
      for (int n = 0; n < H; ++n) {
        double acc = 0.0;
        for (int k = 0; k < I; ++k) {
          acc += rnd(act_ref[(std::size_t)p * I + k]) *
                 deq(qdn, sdn, bdn, r0 + n, I, k);
        }
        part_ref[(std::size_t)p * H + n] = acc;
      }
    }
    const auto c_sgu = conv(sgu), c_bgu = conv(bgu), c_sdn = conv(sdn),
               c_bdn = conv(bdn), c_x = conv(xb);
    SharedBuffer bsgu = upload(c_sgu.data(), c_sgu.size() * 2),
                 bbgu = upload(c_bgu.data(), c_bgu.size() * 2),
                 bsdn = upload(c_sdn.data(), c_sdn.size() * 2),
                 bbdn = upload(c_bdn.data(), c_bdn.size() * 2),
                 bx = upload(c_x.data(), c_x.size() * 2);
    struct SortedBufs {
      SharedBuffer srow, sdst, t2e;
    };
    auto sbuf = [&](const Sorted& s) {
      return SortedBufs{upload(s.srow.data(), s.srow.size() * 4),
                        upload(s.sdst.data(), s.sdst.size() * 4),
                        upload(s.t2e.data(), s.t2e.size() * 4)};
    };
    const auto g2 = sbuf(s2), g32 = sbuf(s32);
    const int npad2 = (int)s2.srow.size(), npad32 = (int)s32.srow.size();
    const int mt2 = (int)s2.t2e.size();
    SharedBuffer act = mc->make_shared_buffer((std::size_t)NP * I * 2),
                 part = mc->make_shared_buffer((std::size_t)NP * H * 2),
                 gact2 = mc->make_shared_buffer((std::size_t)npad2 * I * 2),
                 gdout2 = mc->make_shared_buffer((std::size_t)npad2 * H * 2),
                 gact32 = mc->make_shared_buffer((std::size_t)npad32 * I * 2),
                 gpart32 = mc->make_shared_buffer((std::size_t)NP * H * 2);
    const std::string sfx = bf16 ? "_bf16" : "";
    auto lib_v = mc->load_library("affine_qmv" + sfx);
    auto lib_q = mc->load_library("affine_qmm_steel" + sfx);
    auto f_gs = lib_v.function("affine_gather_qmv_swiglu_w5g64");
    auto f_gd = lib_v.function("affine_gather_down_qmv_w5g64");
    auto f_vs = lib_v.function("affine_grouped_swiglu_w5g64");
    auto f_vd = lib_v.function("affine_grouped_down_w5g64");
    auto f_qs = lib_q.function("affine_qmm_grouped_swiglu_w5g64");
    auto f_qd = lib_q.function("affine_qmm_grouped_w5g64");
    const bool ok = f_gs.valid() && f_gd.valid() && f_vs.valid() &&
                    f_vd.valid() && f_qs.valid() && f_qd.valid();
    ASSERT_TRUE(ok);
    if (!ok) { return; }
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder enc = st.begin_compute();
      // Per-pair gathers.
      enc.set_function(f_gs);
      enc.set_buffer(0, bwgu); enc.set_buffer(1, bsgu);
      enc.set_buffer(2, bbgu); enc.set_buffer(3, bx);
      enc.set_buffer(4, act);
      enc.set_constant(5, H); enc.set_constant(6, 2 * I);
      enc.set_buffer(7, beid); enc.set_constant(8, TOPK);
      enc.dispatch({32, (unsigned)(I / 2), (unsigned)NP}, {32, 2, 1});
      enc.set_function(f_gd);
      enc.set_buffer(0, bwdn); enc.set_buffer(1, bsdn);
      enc.set_buffer(2, bbdn); enc.set_buffer(3, act);
      enc.set_buffer(4, part);
      enc.set_constant(5, I); enc.set_constant(6, H);
      enc.set_buffer(7, beid);
      enc.dispatch({32, (unsigned)(H / 4), (unsigned)NP}, {32, 2, 1});
      // Grouped GEMV (2-row tiles); down writes by sorted slot.
      enc.set_function(f_vs);
      enc.set_buffer(0, bwgu); enc.set_buffer(1, bsgu);
      enc.set_buffer(2, bbgu); enc.set_buffer(3, bx);
      enc.set_buffer(4, gact2);
      enc.set_constant(5, H); enc.set_constant(6, 2 * I);
      enc.set_buffer(7, g2.srow); enc.set_buffer(8, g2.t2e);
      enc.dispatch({32, (unsigned)(I / 2), (unsigned)mt2}, {32, 2, 1});
      enc.set_function(f_vd);
      enc.set_buffer(0, bwdn); enc.set_buffer(1, bsdn);
      enc.set_buffer(2, bbdn); enc.set_buffer(3, gact2);
      enc.set_buffer(4, gdout2);
      enc.set_constant(5, I); enc.set_constant(6, H);
      enc.set_buffer(7, g2.srow); enc.set_buffer(8, g2.t2e);
      enc.dispatch({32, (unsigned)(H / 4), (unsigned)mt2}, {32, 2, 1});
      // Steel grouped GEMMs (32-row tiles); down scatters by sdst.
      enc.set_function(f_qs);
      enc.set_buffer(0, bwgu); enc.set_buffer(1, bsgu);
      enc.set_buffer(2, bbgu); enc.set_buffer(3, bx);
      enc.set_buffer(4, gact32);
      enc.set_constant(5, H); enc.set_constant(6, 2 * I);
      enc.set_constant(7, npad32); enc.set_buffer(8, g32.t2e);
      enc.set_buffer(9, g32.srow);
      enc.dispatch({(unsigned)(((2 * I + 31) / 32) * 32),
                    (unsigned)(((npad32 + 31) / 32) * 2), 2}, {32, 2, 2});
      enc.set_function(f_qd);
      enc.set_buffer(0, bwdn); enc.set_buffer(1, bsdn);
      enc.set_buffer(2, bbdn); enc.set_buffer(3, gact32);
      enc.set_buffer(4, gpart32);
      enc.set_constant(5, I); enc.set_constant(6, H);
      enc.set_constant(7, npad32); enc.set_buffer(8, g32.t2e);
      enc.set_buffer(9, g32.sdst);
      enc.dispatch({(unsigned)(((H + 31) / 32) * 32),
                    (unsigned)(((npad32 + 31) / 32) * 2), 2}, {32, 2, 2});
    }
    ASSERT_TRUE(st.commit().wait_ok());
    // rel-L2 of a [pairs, W] output against its reference; `slot` maps a
    // pair to the row it was written at (sorted outputs), or is empty.
    auto rel = [&](const std::vector<double>& ref, const SharedBuffer& got,
                   int W, const std::vector<std::int32_t>* sdst) {
      const auto* g = static_cast<const std::uint16_t*>(got.contents());
      std::vector<int> row_of(NP);
      for (int p = 0; p < NP; ++p) { row_of[p] = p; }
      if (sdst != nullptr) {
        for (std::size_t k = 0; k < sdst->size(); ++k) {
          if ((*sdst)[k] >= 0) { row_of[(*sdst)[k]] = (int)k; }
        }
      }
      double num = 0.0, den = 0.0;
      for (int p = 0; p < NP; ++p) {
        for (int c = 0; c < W; ++c) {
          const double r = ref[(std::size_t)p * W + c];
          const double v = rd(g[(std::size_t)row_of[p] * W + c]);
          num += (v - r) * (v - r);
          den += r * r;
        }
      }
      return std::sqrt(num / den);
    };
    const double ea = rel(act_ref, act, I, nullptr);
    const double ep = rel(part_ref, part, H, nullptr);
    const double va = rel(act_ref, gact2, I, &s2.sdst);
    const double vp = rel(part_ref, gdout2, H, &s2.sdst);
    const double qa = rel(act_ref, gact32, I, &s32.sdst);
    const double qp = rel(part_ref, gpart32, H, nullptr);
    std::printf("[qwen4_exp] w5 experts vs f64 (%s): gather act %.2e part "
                "%.2e, grouped GEMV %.2e / %.2e, steel %.2e / %.2e\n",
                bf16 ? "bf16" : "f16", ea, ep, va, vp, qa, qp);
    // Storage rounding of act / part is ~2e-3 at bf16 and ~3e-4 at f16; a
    // wrong code, slab or lane is O(1).
    const double bar = bf16 ? 6e-3 : 1.5e-3;
    EXPECT_TRUE(ea < bar && ep < bar);
    EXPECT_TRUE(va < bar && vp < bar);
    EXPECT_TRUE(qa < bar && qp < bar);
  }
}

namespace {

template <typename T>
std::vector<T>
read_raw_(const std::string& path)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { return {}; }
  const std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<T> v((std::size_t)n / sizeof(T));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

double
rel_l2_(const float* ref, const float* got, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)got[i] - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * ref[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

}  // namespace

// The batched w8 GEMV computes each row as the one-row qmv does -- what a
// speculative verify's few rows rest on -- at the hyper-connection model's
// widths, a partial last block (K 320, 640) included.
TEST(qwen4_exp_kernels, batched_w8_gemv_rows_are_qmv_rows)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  using metal_compute::ComputeEncoder;
  using metal_compute::SharedBuffer;
  std::uint32_t rng = 7u;
  auto next = [&]() {
    rng = rng * 1664525u + 1013904223u;
    return rng;
  };
  auto unif = [&]() { return (next() >> 8) * (1.0 / 16777216.0); };
  for (const bool bf16 : {false, true}) {
    const std::string sfx = bf16 ? "_bf16" : "";
    auto lib = mc->load_library("affine_qmv" + sfx);
    auto f1 = lib.function("affine_qmv_w8g64");
    auto f4 = lib.function("affine_qmv_batch4_w8g64");
    ASSERT_TRUE(f1.valid() && f4.valid());
    if (!f1.valid() || !f4.valid()) { return; }
    for (const auto [K, N] : {std::pair{10240, 320}, std::pair{320, 10240},
                              std::pair{640, 2560}, std::pair{2560, 16480}}) {
      constexpr int M = 3;
      auto el = [&](double d) {
        return bf16 ? to_bf16_((float)d) : to_f16_(d);
      };
      std::vector<std::uint8_t> q((std::size_t)N * K);
      for (auto& v : q) { v = (std::uint8_t)(next() >> 24); }
      std::vector<std::uint16_t> sc((std::size_t)N * K / 64),
          bi((std::size_t)N * K / 64), x((std::size_t)M * K);
      for (auto& v : sc) { v = el(0.001 + 0.01 * unif()); }
      for (auto& v : bi) { v = el(-0.5 * unif()); }
      for (auto& v : x) { v = el(2.0 * unif() - 1.0); }
      auto up = [&](const void* p, std::size_t n) {
        SharedBuffer b = mc->make_shared_buffer(n);
        std::memcpy(b.contents(), p, n);
        return b;
      };
      SharedBuffer bw = up(q.data(), q.size());
      SharedBuffer bs = up(sc.data(), sc.size() * 2);
      SharedBuffer bb = up(bi.data(), bi.size() * 2);
      SharedBuffer bx = up(x.data(), x.size() * 2);
      SharedBuffer y1 = mc->make_shared_buffer((std::size_t)M * N * 2);
      SharedBuffer y4 = mc->make_shared_buffer((std::size_t)M * N * 2);
      auto st = mc->make_command_stream();
      {
        ComputeEncoder enc = st.begin_compute();
        for (int r = 0; r < M; ++r) {
          enc.set_function(f1);
          enc.set_buffer(0, bw); enc.set_buffer(1, bs); enc.set_buffer(2, bb);
          enc.set_buffer(3, bx, (std::size_t)r * K * 2);
          enc.set_buffer(4, y1, (std::size_t)r * N * 2);
          enc.set_constant(5, K); enc.set_constant(6, N);
          enc.dispatch({32, (unsigned)((N + 3) / 4), 1}, {32, 2, 1});
        }
        enc.set_function(f4);
        enc.set_buffer(0, bw); enc.set_buffer(1, bs); enc.set_buffer(2, bb);
        enc.set_buffer(3, bx); enc.set_buffer(4, y4);
        enc.set_constant(5, K); enc.set_constant(6, N);
        enc.set_constant(7, M);
        enc.dispatch({32, (unsigned)((N + 3) / 4), 1}, {32, 2, 1});
      }
      ASSERT_TRUE(st.commit().wait_ok());
      const auto* a = static_cast<const std::uint16_t*>(y1.contents());
      const auto* b = static_cast<const std::uint16_t*>(y4.contents());
      int diff = 0;
      for (std::size_t i = 0; i < (std::size_t)M * N; ++i) {
        diff += a[i] != b[i];
      }
      std::printf("[qwen4_exp_kernels] w8 GEMV %s K %d N %d: %d of %d "
                  "outputs differ between qmv rows and the batched one\n",
                  bf16 ? "bf16" : "f16", K, N, diff, M * N);
      EXPECT_TRUE(diff == 0);
    }
  }
}

// The sparse rows' attention, every kernel a row can take: the per-head
// one (a decode's rows), the key-split one, and a prefill's group kernels
// (simdgroup MMA; the matrix units' on an M5) -- over each row's selected
// tokens gathered from a paged pool whose pages are out of order, at f16
// and bf16, against f64 on the same rounded inputs; the per-head kernel
// keeps P in f32, so its distance is the output's rounding and the others
// are held to it. The values carry a common offset, as attention's do, so
// a row's scale error shows -- though not finely: the group kernels round
// P to the element type and divide by the sum of THOSE weights, and
// dividing by the f32 sum instead reads only ~1% further here (M4 Pro,
// M5). G = 12 is Flash-Next's (two 8-row bands overlapping), G = 8 one
// band. MEASURED (both boxes): bf16 1.222e-3 per-head, 1.239e-3 group.
TEST(qwen4_exp_kernels, qsa_attention_matches_f64)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  using metal_compute::ComputeEncoder;
  using metal_compute::SharedBuffer;
  std::uint32_t rng = 11u;
  auto next = [&]() {
    rng = rng * 1664525u + 1013904223u;
    return rng;
  };
  auto unif = [&]() { return (next() >> 8) * (1.0 / 16777216.0); };
  constexpr int D = 256, Hkv = 2, PT = 256, NP = 12, T = PT * NP;
  constexpr int W = 2112;                   // a row's selection width
  const std::vector<int> nsel = {2051, 1203, 37, 1, 2112};
  const int n = (int)nsel.size(), r0 = 0;
  const float scale = 1.0f / 16.0f;
  const bool nax = mc->supports_matrix_cores();
  for (const bool bf16 : {false, true}) {
    const std::string sfx = bf16 ? "_bf16" : "";
    auto lhc = mc->load_library("qwen4_hc" + sfx);
    auto lmm = mc->load_library("sdpa_mma" + sfx);
    struct Arm { const char* name; metal_compute::ComputeFunction fn; };
    std::vector<Arm> arms;
    arms.push_back({"per-head", lhc.function("qsa_attn_f16")});
    arms.push_back({"key-split", lhc.function("qsa_attn_split_f16")});
    arms.push_back({"group mma", lhc.function("qsa_attn_mma_f16")});
    if (nax) {
      arms.push_back({"group nax", lmm.function("qsa_attn_nax_f16")});
    }
    for (const auto& a : arms) { ASSERT_TRUE(a.fn.valid()); }
    auto el = [&](double d) {
      return bf16 ? to_bf16_((float)d) : to_f16_(d);
    };
    auto de = [&](std::uint16_t u) {
      return bf16 ? from_bf16_(u) : from_f16_(u);
    };
    for (const int G : {12, 8}) {
      const int Hq = G * Hkv;
      std::vector<std::uint16_t> q((std::size_t)Hq * n * D),
          k((std::size_t)NP * Hkv * PT * D), v(k.size());
      for (auto& x : q) { x = el(6.0 * (2.0 * unif() - 1.0)); }
      for (auto& x : k) { x = el(2.0 * unif() - 1.0); }
      for (auto& x : v) { x = el(0.4 + (2.0 * unif() - 1.0)); }
      // Logical page i lives in physical page 5i mod 12.
      std::vector<int> pt((std::size_t)NP * 3);
      for (int i = 0; i < NP; ++i) {
        pt[(std::size_t)i * 3 + 0] = (5 * i) % NP;
        pt[(std::size_t)i * 3 + 1] = PT;
        pt[(std::size_t)i * 3 + 2] = i * PT;
      }
      std::vector<int> sel((std::size_t)n * W, 0);
      std::vector<int> perm(T);
      for (int r = 0; r < n; ++r) {
        for (int t = 0; t < T; ++t) { perm[(std::size_t)t] = t; }
        for (int t = T - 1; t > 0; --t) {
          std::swap(perm[(std::size_t)t],
                    perm[(std::size_t)(next() % (std::uint32_t)(t + 1))]);
        }
        for (int j = 0; j < nsel[(std::size_t)r]; ++j) {
          sel[(std::size_t)r * W + j] = perm[(std::size_t)j];
        }
      }
      auto at = [&](int pos, int kvh) {
        const int lp = pos / PT, slot = pos % PT;
        const int pg = pt[(std::size_t)lp * 3];
        return (((std::size_t)pg * Hkv + kvh) * PT + slot) * D;
      };
      // f64 on the rounded inputs.
      std::vector<double> ref((std::size_t)Hq * n * D, 0.0);
      for (int h = 0; h < Hq; ++h) {
        for (int r = 0; r < n; ++r) {
          const int m = nsel[(std::size_t)r], kvh = h / G;
          const std::uint16_t* qr = &q[((std::size_t)h * n + r) * D];
          std::vector<double> sc((std::size_t)m);
          double mx = -1e300;
          for (int j = 0; j < m; ++j) {
            const std::size_t o = at(sel[(std::size_t)r * W + j], kvh);
            double dot = 0.0;
            for (int i = 0; i < D; ++i) {
              dot += de(qr[i]) * de(k[o + (std::size_t)i]);
            }
            sc[(std::size_t)j] = dot * (double)scale;
            mx = std::max(mx, sc[(std::size_t)j]);
          }
          double sum = 0.0;
          for (auto& x : sc) { x = std::exp(x - mx); sum += x; }
          double* orow = &ref[((std::size_t)h * n + r) * D];
          for (int j = 0; j < m; ++j) {
            const std::size_t o = at(sel[(std::size_t)r * W + j], kvh);
            for (int i = 0; i < D; ++i) {
              orow[i] += sc[(std::size_t)j] / sum * de(v[o + (std::size_t)i]);
            }
          }
        }
      }
      auto up = [&](const void* p, std::size_t bytes) {
        SharedBuffer b = mc->make_shared_buffer(bytes);
        std::memcpy(b.contents(), p, bytes);
        return b;
      };
      SharedBuffer bq = up(q.data(), q.size() * 2);
      SharedBuffer bk = up(k.data(), k.size() * 2);
      SharedBuffer bv = up(v.data(), v.size() * 2);
      SharedBuffer bpt = up(pt.data(), pt.size() * 4);
      SharedBuffer bsel = up(sel.data(), sel.size() * 4);
      SharedBuffer bns = up(nsel.data(), nsel.size() * 4);
      double err_head = 0.0;
      for (const auto& a : arms) {
        SharedBuffer bo = mc->make_shared_buffer(q.size() * 2);
        std::memset(bo.contents(), 0xFF, q.size() * 2);   // NaN if unwritten
        auto st = mc->make_command_stream();
        {
          ComputeEncoder enc = st.begin_compute();
          enc.set_function(a.fn);
          enc.set_buffer(0, bq); enc.set_buffer(1, bk);
          enc.set_buffer(2, bv); enc.set_buffer(3, bo);
          enc.set_buffer(4, bpt); enc.set_buffer(5, bsel);
          enc.set_buffer(6, bns);
          enc.set_constant(7, scale); enc.set_constant(8, D);
          enc.set_constant(9, Hq); enc.set_constant(10, Hkv);
          enc.set_constant(11, n); enc.set_constant(12, PT);
          enc.set_constant(13, NP); enc.set_constant(14, W);
          enc.set_constant(15, r0);
          const std::string nm = a.name;
          if (nm == "group mma" || nm == "group nax") {
            enc.dispatch({256, (unsigned)Hkv, (unsigned)n}, {256, 1, 1});
          } else if (nm == "key-split") {
            enc.dispatch({32, 16u * (unsigned)Hq, (unsigned)n}, {32, 16, 1});
          } else {
            enc.dispatch({32, (unsigned)Hq, (unsigned)n}, {32, 1, 1});
          }
        }
        ASSERT_TRUE(st.commit().wait_ok());
        const auto* o = static_cast<const std::uint16_t*>(bo.contents());
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
          const double d = de(o[i]) - ref[i];
          num += d * d;
          den += ref[i] * ref[i];
        }
        const double e = std::sqrt(num / den);
        if (std::string(a.name) == "per-head") { err_head = e; }
        std::printf("[qwen4_exp_kernels] QSA attention %s G %d %s: rel-L2 "
                    "%.3e from f64\n", bf16 ? "bf16" : "f16", G, a.name, e);
        // The per-head kernel keeps P in f32: its error is the output's
        // rounding. The others within 1.5x of it.
        EXPECT_TRUE(std::isfinite(e));
        EXPECT_TRUE(e < (bf16 ? 2.5e-3 : 4e-4));
        EXPECT_TRUE(err_head > 0.0 && e < 1.5 * err_head);
      }
    }
  }
}

// A teacher-forced prefill over the golden's prompt + continuation (86
// rows: every layer kind, the PLE layer, every expert route the grouped
// GEMMs and the cache take) against the fp32 reference: every row's
// logits, held to the reference's own bf16 distance, and every DECISIVE
// row's argmax (fp32 top-2 margin > 0.5) the reference's.
// QSA'S SELECT IN TWO PHASES (qsa_topk_chunk / _merge: chunks of 4096
// blocks each their own top k, then a row's of theirs) against the one
// threadgroup's (qsa_select): the same tokens and counts, row for row, at
// 1 to 16 chunks -- scores on a coarse grid so ties sit at the threshold,
// a verify's three rows each seeing its own blocks (the rows past one's
// blocks read -1) -- and each one's time a decode row.
TEST(qwen4_exp_kernels, qsa_two_phase_select_picks_the_same)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto lhc = mc->load_library("qwen4_hc");
  auto one = lhc.function("qsa_select_f16");
  auto chunk = lhc.function("qsa_topk_chunk_f16");
  auto merge = lhc.function("qsa_topk_merge_f16");
  ASSERT_TRUE(one.valid() && chunk.valid() && merge.valid());
  if (!one.valid() || !chunk.valid() || !merge.valid()) { return; }
  constexpr int ratio = 4, k = 512, CH = 4096;
  const int width = k * ratio + ratio - 1;
  std::uint32_t rng = 5u;
  auto next = [&]() {
    rng = rng * 1664525u + 1013904223u;
    return rng;
  };
  auto stream = mc->make_command_stream();
  for (const int rows : {1, 3}) {
    for (const int nb0 : {3000, 9000, 32768, 40001, 65536}) {
      const int r0 = 0;
      const int q_offset = nb0 * ratio - 1 - (rows - 1);
      const int nbm = (q_offset + r0 + rows) / ratio;
      std::vector<float> sc((std::size_t)rows * nbm);
      for (int r = 0; r < rows; ++r) {
        const int nb = (q_offset + r0 + r + 1) / ratio;
        for (int b = 0; b < nbm; ++b) {
          sc[(std::size_t)r * nbm + b] =
              b < nb ? (float)(next() % 4096) / 64.0f : -1.0f;
        }
      }
      auto s = mc->make_shared_buffer(sc.size() * 4);
      std::memcpy(s.contents(), sc.data(), sc.size() * 4);
      const int G = (nbm + CH - 1) / CH;
      const int R = std::clamp(3 * k / 2 / G, 1, std::min(k, 3072 / G));
      auto sel1 = mc->make_shared_buffer((std::size_t)rows * width * 4);
      auto ns1 = mc->make_shared_buffer((std::size_t)rows * 4);
      auto sel2 = mc->make_shared_buffer((std::size_t)rows * width * 4);
      auto ns2 = mc->make_shared_buffer((std::size_t)rows * 4);
      auto ck = mc->make_shared_buffer((std::size_t)rows * G * k * 4);
      auto cp = mc->make_shared_buffer((std::size_t)rows * G * k * 4);
      auto tk = mc->make_shared_buffer((std::size_t)rows * G * 4);
      auto tr = mc->make_shared_buffer((std::size_t)rows * G * R * 4);
      std::memset(sel1.contents(), 0xff, sel1.byte_size());
      std::memset(sel2.contents(), 0xee, sel2.byte_size());
      auto enc_one = [&](metal_compute::ComputeEncoder& enc) {
        enc.set_function(one);
        enc.set_buffer(0, s); enc.set_buffer(1, sel1);
        enc.set_buffer(2, ns1);
        enc.set_constant(3, nbm); enc.set_constant(4, k);
        enc.set_constant(5, ratio); enc.set_constant(6, q_offset);
        enc.set_constant(7, r0);
        enc.dispatch({256u * (unsigned)rows, 1, 1}, {256, 1, 1});
      };
      auto enc_two = [&](metal_compute::ComputeEncoder& enc) {
        enc.set_function(chunk);
        enc.set_buffer(0, s); enc.set_buffer(1, ck); enc.set_buffer(2, cp);
        enc.set_buffer(3, tk); enc.set_buffer(4, tr);
        enc.set_constant(5, nbm); enc.set_constant(6, k);
        enc.set_constant(7, R); enc.set_constant(8, ratio);
        enc.set_constant(9, q_offset); enc.set_constant(10, r0);
        enc.dispatch({256u * (unsigned)G, (unsigned)rows, 1}, {256, 1, 1});
        enc.set_function(merge);
        enc.set_buffer(0, ck); enc.set_buffer(1, cp); enc.set_buffer(2, tk);
        enc.set_buffer(3, tr); enc.set_buffer(4, sel2);
        enc.set_buffer(5, ns2);
        enc.set_constant(6, nbm); enc.set_constant(7, k);
        enc.set_constant(8, R); enc.set_constant(9, ratio);
        enc.set_constant(10, q_offset); enc.set_constant(11, r0);
        enc.dispatch({256u * (unsigned)rows, 1, 1}, {256, 1, 1});
      };
      {
        auto enc = stream.begin_compute();
        enc_one(enc);
        enc_two(enc);
      }
      ASSERT_TRUE(stream.commit().wait_ok());
      const auto* a = static_cast<const std::int32_t*>(sel1.contents());
      const auto* b = static_cast<const std::int32_t*>(sel2.contents());
      const auto* na = static_cast<const std::int32_t*>(ns1.contents());
      const auto* nb2 = static_cast<const std::int32_t*>(ns2.contents());
      int bad = 0;
      for (int r = 0; r < rows; ++r) {
        if (na[r] != nb2[r]) { ++bad; continue; }
        for (int i = 0; i < na[r]; ++i) {
          bad += a[(std::size_t)r * width + i] !=
                 b[(std::size_t)r * width + i];
        }
      }
      // a decode row's time each way (rows 1)
      double us[2] = {0.0, 0.0};
      if (rows == 1) {
        for (int arm = 0; arm < 2; ++arm) {
          constexpr int N = 100;
          const auto t0 = std::chrono::steady_clock::now();
          {
            auto enc = stream.begin_compute();
            for (int i = 0; i < N; ++i) {
              if (arm == 0) { enc_one(enc); } else { enc_two(enc); }
            }
          }
          ASSERT_TRUE(stream.commit().wait_ok());
          us[arm] = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - t0).count() / N;
        }
      }
      std::printf("[qwen4_exp_kernels] QSA select, %d row(s), %d blocks (%d "
                  "chunks): %d tokens differ%s", rows, nbm, G, bad,
                  rows == 1 ? "" : "\n");
      if (rows == 1) {
        std::printf("; one threadgroup %.1f us, two phases %.1f us\n", us[0],
                    us[1]);
      }
      EXPECT_TRUE(bad == 0);
    }
  }
}

TEST(qwen4_exp, prefill_matches_reference)
{
  const char* mp = std::getenv("VPIPE_QWEN4_EXP_TEST_MODEL_PATH");
  const char* gp = std::getenv("VPIPE_QWEN4_EXP_GOLDEN");
  if (mp == nullptr || gp == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  using vpipe::genai::MetalQwenModel;
  vpipe::genai::ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(mp);
  ASSERT_TRUE(mcfg.has_value());
  if (!mcfg.has_value()) { return; }
  MetalQwenModel::Config c = MetalQwenModel::config_from(*mcfg);
  ASSERT_TRUE(c.hc() && c.ple_layer == 1 && c.n_experts == 512);
  c.use_bf16 = std::getenv("VPIPE_QWEN4_EXP_F16") == nullptr;
  c.page_tokens = 256;
  c.max_pages = 64;
  c.max_seq = 8192;
  std::size_t cache_mb = 36864;
  if (const char* e = std::getenv("VPIPE_QWEN4_EXP_CACHE_MB")) {
    cache_mb = (std::size_t)std::atoll(e);
  }
  c.expert_cache_bytes = cache_mb << 20;
  const std::string g = gp;
  const auto ids = read_raw_<std::int32_t>(g + "/ids.i32");
  const auto ref = read_raw_<float>(g + "/logits_f32.f32");
  const auto ref16 = read_raw_<float>(g + "/logits_bf16.f32");
  const std::size_t T = ids.size();
  ASSERT_TRUE(T > 0 && ref.size() % T == 0 && ref16.size() == ref.size());
  if (T == 0 || ref.size() % T != 0) { return; }
  const std::size_t V = ref.size() / T;
  auto m = MetalQwenModel::load(mp, mc, c);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<float> got = m->prefill_all_logits(
      m->root_context(), std::vector<std::int32_t>(ids.begin(), ids.end()));
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(got.size() == ref.size());
  if (got.size() != ref.size()) { return; }
  int decisive = 0, agree = 0, agree_all = 0;
  for (std::size_t t = 0; t < T; ++t) {
    const float* r = ref.data() + t * V;
    const float* o = got.data() + t * V;
    std::size_t a = 0, b = 0, ao = 0;
    for (std::size_t v = 1; v < V; ++v) {
      if (r[v] > r[a]) { a = v; }
      if (o[v] > o[ao]) { ao = v; }
    }
    b = a == 0 ? 1 : 0;
    for (std::size_t v = 0; v < V; ++v) {
      if (v != a && r[v] > r[b]) { b = v; }
    }
    agree_all += ao == a;
    if (r[a] - r[b] > 0.5f) {
      ++decisive;
      agree += ao == a;
    }
  }
  const double e = rel_l2_(ref.data(), got.data(), ref.size());
  const double bar = rel_l2_(ref.data(), ref16.data(), ref.size());
  std::printf("[qwen4_exp] prefill %zu rows (%s) in %.0f ms: logits %.4f "
              "from fp32 (the reference's bf16 %.4f); argmax %d/%zu, "
              "decisive %d/%d\n", T, c.use_bf16 ? "bf16" : "f16", ms, e, bar,
              agree_all, T, agree, decisive);
  EXPECT_TRUE(e < 1.5 * bar);
  EXPECT_TRUE(agree == decisive);
}

// The decode path over the same rows: the golden's first P rows prefilled,
// then every following row a one-token step (forward(): n = 1, the PLE's
// hashed history and conv state, the GDN states and the KV carried between
// calls), each step's logits against the reference's row.
TEST(qwen4_exp, decode_matches_reference)
{
  const char* mp = std::getenv("VPIPE_QWEN4_EXP_TEST_MODEL_PATH");
  const char* gp = std::getenv("VPIPE_QWEN4_EXP_GOLDEN");
  if (mp == nullptr || gp == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  using vpipe::genai::MetalQwenModel;
  vpipe::genai::ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(mp);
  ASSERT_TRUE(mcfg.has_value());
  if (!mcfg.has_value()) { return; }
  MetalQwenModel::Config c = MetalQwenModel::config_from(*mcfg);
  c.use_bf16 = std::getenv("VPIPE_QWEN4_EXP_F16") == nullptr;
  c.page_tokens = 256;
  c.max_pages = 64;
  c.max_seq = 8192;
  std::size_t cache_mb = 36864;
  if (const char* e = std::getenv("VPIPE_QWEN4_EXP_CACHE_MB")) {
    cache_mb = (std::size_t)std::atoll(e);
  }
  c.expert_cache_bytes = cache_mb << 20;
  const std::string g = gp;
  const auto ids = read_raw_<std::int32_t>(g + "/ids.i32");
  const auto ref = read_raw_<float>(g + "/logits_f32.f32");
  const auto ref16 = read_raw_<float>(g + "/logits_bf16.f32");
  const std::size_t T = ids.size();
  if (T < 8 || ref.size() % T != 0) { return; }
  const std::size_t V = ref.size() / T;
  const std::size_t P = T - 24;            // the last 24 rows step by step
  auto m = MetalQwenModel::load(mp, mc, c);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  const auto cid = m->root_context();
  std::vector<float> step = m->prefill(
      cid, std::vector<std::int32_t>(ids.begin(), ids.begin() + (long)P));
  ASSERT_TRUE(step.size() == V);
  std::vector<float> got((T - P + 1) * V);
  std::memcpy(got.data(), step.data(), V * 4);   // row P - 1
  const auto t0 = std::chrono::steady_clock::now();
  for (std::size_t t = P; t < T; ++t) {
    step = m->forward(cid, ids[t]);
    ASSERT_TRUE(step.size() == V);
    if (step.size() != V) { return; }
    std::memcpy(got.data() + (t - P + 1) * V, step.data(), V * 4);
  }
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
  const float* r0 = ref.data() + (P - 1) * V;
  const float* b0 = ref16.data() + (P - 1) * V;
  const std::size_t nr = (T - P + 1) * V;
  int agree = 0;
  for (std::size_t t = 0; t <= T - P; ++t) {
    const float* r = r0 + t * V;
    const float* o = got.data() + t * V;
    std::size_t a = 0, ao = 0;
    for (std::size_t v = 1; v < V; ++v) {
      if (r[v] > r[a]) { a = v; }
      if (o[v] > o[ao]) { ao = v; }
    }
    agree += ao == a;
  }
  const double e = rel_l2_(r0, got.data(), nr);
  const double bar = rel_l2_(r0, b0, nr);
  std::printf("[qwen4_exp] %zu-row prefill + %zu decode steps (%s, %.0f ms "
              "a step): logits %.4f from fp32 (the reference's bf16 %.4f); "
              "argmax %d/%zu\n", P, T - P, c.use_bf16 ? "bf16" : "f16",
              ms / (double)(T - P), e, bar, agree, T - P + 1);
  EXPECT_TRUE(e < 1.5 * bar);
  EXPECT_TRUE(agree >= (int)(T - P + 1) - 1);
}

// The Lightning MTP head, teacher-forced as a prefill primes it: each
// (residual before the final mixer at row t, token t + 1) pair through
// the head's fuse, its one hyper-connection layer over its own KV, its
// mixer and the shared head -- every pair's logits (they predict token
// t + 2) against the reference's (the dumper's --mtp-from run), held to
// its bf16 distance, and every decisive row's argmax. Decisive is a fp32
// margin over 1.0 here: the head reads a residual 48 bf16 layers deep
// (0.13 from fp32 by then), and in bf16 two rows of the golden's at 0.68 /
// 0.74 turn (one an exact bf16 tie); in f16 every row over 0.5 holds.
TEST(qwen4_exp, mtp_head_matches_reference)
{
  const char* mp = std::getenv("VPIPE_QWEN4_EXP_TEST_MODEL_PATH");
  const char* gp = std::getenv("VPIPE_QWEN4_EXP_GOLDEN");
  if (mp == nullptr || gp == nullptr) { return; }
  const std::string g = gp;
  const auto ref = read_raw_<float>(g + "/mtp_logits_f32.f32");
  const auto ref16 = read_raw_<float>(g + "/mtp_logits_bf16.f32");
  if (ref.empty()) { return; }          // a golden without the head's
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  using vpipe::genai::MetalQwenModel;
  vpipe::genai::ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(mp);
  ASSERT_TRUE(mcfg.has_value());
  if (!mcfg.has_value()) { return; }
  MetalQwenModel::Config c = MetalQwenModel::config_from(*mcfg);
  c.use_bf16 = std::getenv("VPIPE_QWEN4_EXP_F16") == nullptr;
  c.page_tokens = 256;
  c.max_pages = 64;
  c.max_seq = 8192;
  std::size_t cache_mb = 36864;
  if (const char* e = std::getenv("VPIPE_QWEN4_EXP_CACHE_MB")) {
    cache_mb = (std::size_t)std::atoll(e);
  }
  c.expert_cache_bytes = cache_mb << 20;
  const auto ids = read_raw_<std::int32_t>(g + "/ids.i32");
  const std::size_t T = ids.size();
  ASSERT_TRUE(T > 2 && ref.size() % (T - 1) == 0 &&
              ref16.size() == ref.size());
  if (T <= 2 || ref.size() % (T - 1) != 0) { return; }
  const std::size_t V = ref.size() / (T - 1);
  auto m = MetalQwenModel::load(mp, mc, c);
  ASSERT_TRUE(m != nullptr && m->has_mtp());
  if (m == nullptr || !m->has_mtp()) { return; }
  std::vector<float> got;
  const std::vector<float> all = m->prefill_all_logits(
      m->root_context(), std::vector<std::int32_t>(ids.begin(), ids.end()),
      &got);
  ASSERT_TRUE(!all.empty() && got.size() == ref.size());
  if (got.size() != ref.size()) { return; }
  int decisive = 0, agree = 0, agree16 = 0, hits = 0, hits_ref = 0;
  for (std::size_t t = 0; t + 1 < T; ++t) {
    const float* r = ref.data() + t * V;
    const float* o = got.data() + t * V;
    const float* b16 = ref16.data() + t * V;
    std::size_t a = 0, ao = 0, a16 = 0;
    for (std::size_t v = 1; v < V; ++v) {
      if (r[v] > r[a]) { a = v; }
      if (o[v] > o[ao]) { ao = v; }
      if (b16[v] > b16[a16]) { a16 = v; }
    }
    std::size_t b = a == 0 ? 1 : 0;
    for (std::size_t v = 0; v < V; ++v) {
      if (v != a && r[v] > r[b]) { b = v; }
    }
    if (t + 2 < T) {
      hits += (int)ao == ids[t + 2];
      hits_ref += (int)a == ids[t + 2];
    }
    if (r[a] - r[b] > (c.use_bf16 ? 1.0f : 0.5f)) {
      ++decisive;
      agree += ao == a;
      agree16 += a16 == a;
      if (ao != a) {
        std::printf("[qwen4_exp]   pair %zu missed: fp32 margin %.3f, ours "
                    "%.3f over its pick\n", t, r[a] - r[b], o[ao] - o[a]);
      }
    }
  }
  const double e = rel_l2_(ref.data(), got.data(), ref.size());
  const double bar = rel_l2_(ref.data(), ref16.data(), ref.size());
  std::printf("[qwen4_exp] MTP head over %zu pairs (%s): logits %.4f from "
              "fp32 (the reference's bf16 %.4f); decisive argmax %d/%d (its "
              "bf16 %d); drafts that hit the token two ahead %d/%zu (fp32 "
              "%d)\n", T - 1, c.use_bf16 ? "bf16" : "f16", e, bar, agree,
              decisive, agree16, hits, T - 2, hits_ref);
  EXPECT_TRUE(e < 1.15 * bar);
  EXPECT_TRUE(agree >= agree16);
}

// Speculative decode with the Lightning head keeps the plain decode's
// tokens: the golden's prompt prefilled (the head primed with it), then N
// tokens by the pipelined decode on one context and by mtp_decode on
// another -- greedy, then sampled with penalties -- the same tokens, and
// drafts accepted. Prints the speed against a warm plain decode
// (VPIPE_QWEN4_EXP_MTP_COLD=1: the speculative one on the cold cache).
TEST(qwen4_exp, mtp_decode_keeps_the_plain_tokens)
{
  const char* mp = std::getenv("VPIPE_QWEN4_EXP_TEST_MODEL_PATH");
  const char* gp = std::getenv("VPIPE_QWEN4_EXP_GOLDEN");
  if (mp == nullptr || gp == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  using vpipe::genai::MetalQwenModel;
  vpipe::genai::ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(mp);
  ASSERT_TRUE(mcfg.has_value());
  if (!mcfg.has_value()) { return; }
  MetalQwenModel::Config c = MetalQwenModel::config_from(*mcfg);
  c.use_bf16 = std::getenv("VPIPE_QWEN4_EXP_F16") == nullptr;
  c.page_tokens = 256;
  c.max_pages = 64;
  c.max_seq = 8192;
  std::size_t cache_mb = 36864;
  if (const char* e = std::getenv("VPIPE_QWEN4_EXP_CACHE_MB")) {
    cache_mb = (std::size_t)std::atoll(e);
  }
  c.expert_cache_bytes = cache_mb << 20;
  const auto all = read_raw_<std::int32_t>(std::string(gp) + "/ids.i32");
  if (all.size() < 32) { return; }
  // The prompt: the golden's rows up to its continuation's first words.
  const std::vector<std::int32_t> prompt(all.begin(), all.end() - 20);
  int N = 48;
  if (const char* e = std::getenv("VPIPE_QWEN4_EXP_MTP_TOKENS")) {
    N = std::max(2, std::atoi(e));
  }
  auto m = MetalQwenModel::load(mp, mc, c);
  ASSERT_TRUE(m != nullptr && m->has_mtp());
  if (m == nullptr || !m->has_mtp()) { return; }
  m->set_mtp_prefix_seed(true);
  auto argmax = [](const std::vector<float>& v) {
    std::size_t a = 0;
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > v[a]) { a = i; }
    }
    return (std::int32_t)a;
  };
  using clk = std::chrono::steady_clock;
  // The pipelined greedy decode -- twice, the second after the speculative
  // run, which is the one timed: the expert cache is then as warm for it
  // as for the speculative decode (the first run reads what both select).
  vpipe::genai::GpuSamplerParams gs;
  std::int32_t first = -1;
  auto run_plain = [&](vpipe::genai::ContextId cx, double* ms) {
    first = argmax(m->prefill(cx, prompt));
    std::vector<std::int32_t> out{first};
    const auto t0 = clk::now();
    if (!m->pdecode_begin(cx, first, prompt, gs, N)) { return out; }
    for (int i = 1; i < N; ++i) {
      if (!m->pdecode_commit(cx)) { break; }
      const std::int32_t t = m->pdecode_next(cx);
      if (t < 0) { break; }
      out.push_back(t);
    }
    m->pdecode_end(cx);
    *ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    return out;
  };
  // VPIPE_QWEN4_EXP_MTP_COLD=1 decodes speculatively FIRST, on the cold
  // cache, for the other end of the comparison.
  const bool spec_cold = std::getenv("VPIPE_QWEN4_EXP_MTP_COLD") != nullptr;
  double cold_ms = 0.0, plain_ms = 0.0;
  std::vector<std::int32_t> plain;
  if (!spec_cold) {
    plain = run_plain(m->root_context(), &cold_ms);
    ASSERT_TRUE((int)plain.size() == N);
  }
  // The same prompt on another context, decoded speculatively.
  const auto cb = m->context_manager()->acquire_root();
  ASSERT_TRUE(cb.valid());
  const std::int32_t first_b = argmax(m->prefill(cb, prompt));
  ASSERT_TRUE(spec_cold || first_b == first);
  std::vector<std::int32_t> spec;
  long acc = 0, rounds = 0;
  const auto t1 = clk::now();
  ASSERT_TRUE(m->mtp_decode(cb, first_b, N, spec, 1, &acc, &rounds));
  const double spec_ms =
      std::chrono::duration<double, std::milli>(clk::now() - t1).count();
  const auto cc = m->context_manager()->acquire_root();
  const std::vector<std::int32_t> again = run_plain(cc, &plain_ms);
  if (spec_cold) {
    plain = again;
  } else {
    EXPECT_TRUE(again == plain);
  }
  int same = 0;
  while (same < (int)std::min(spec.size(), plain.size()) &&
         spec[(std::size_t)same] == plain[(std::size_t)same]) {
    ++same;
  }
  if (same < (int)std::min(spec.size(), plain.size())) {
    // Where the two part: the logits there by a fresh prefill of what both
    // agree on -- a near-tie the verify's rows and a decode step's single
    // row round apart, or not.
    const auto cc = m->context_manager()->acquire_root();
    std::vector<std::int32_t> pre = prompt;
    pre.insert(pre.end(), plain.begin(), plain.begin() + same);
    const std::vector<float> lg = m->prefill(cc, pre);
    const std::size_t a = (std::size_t)plain[(std::size_t)same];
    const std::size_t b = (std::size_t)spec[(std::size_t)same];
    std::printf("[qwen4_exp]   parted at token %d: plain %zu, MTP %zu; a "
                "prefill there puts them %.4f / %.4f (argmax %d)\n", same,
                a, b, lg[a], lg[b], argmax(lg));
  }
  std::printf("[qwen4_exp] MTP decode (%s): %zu tokens in %ld rounds (%.2f "
              "a round), %ld drafts kept; %.0f ms a token against the plain "
              "decode's %.0f warm (%.2fx; %.0f cold); the first %d of %zu "
              "tokens the plain decode's\n", c.use_bf16 ? "bf16" : "f16",
              spec.size(), rounds,
              rounds > 0 ? (double)spec.size() / (double)rounds : 0.0, acc,
              spec_ms / (double)spec.size(), plain_ms / (double)plain.size(),
              plain_ms / (double)plain.size() /
                  (spec_ms / (double)spec.size()),
              cold_ms / (double)plain.size(), same, plain.size());
  EXPECT_TRUE(spec.size() == plain.size());
  EXPECT_TRUE(same == (int)plain.size());
  EXPECT_TRUE(acc > 0);
  // SAMPLED, with the chat pipelines' penalties: each verify row samples
  // with the seed and seen-set a decode step would hold there, so the
  // speculative decode keeps the pipelined sampler's tokens too.
  vpipe::genai::GpuSamplerParams ss;
  ss.greedy = false;
  ss.temperature = 0.7f;
  ss.top_k = 20;
  ss.top_p = 0.8f;
  ss.presence_penalty = 1.5f;
  ss.seed = 42;
  const int Ns = std::min(N, 48);
  const auto cd = m->context_manager()->acquire_root();
  const std::int32_t f2 = argmax(m->prefill(cd, prompt));
  std::vector<std::int32_t> ps{f2};
  ASSERT_TRUE(m->pdecode_begin(cd, f2, prompt, ss, Ns));
  for (int i = 1; i < Ns; ++i) {
    if (!m->pdecode_commit(cd)) { break; }
    const std::int32_t t = m->pdecode_next(cd);
    if (t < 0) { break; }
    ps.push_back(t);
  }
  m->pdecode_end(cd);
  const auto ce = m->context_manager()->acquire_root();
  ASSERT_TRUE(argmax(m->prefill(ce, prompt)) == f2);
  vpipe::genai::MtpDecodeCtl sc;
  sc.sampler = ss;
  sc.prime = prompt;
  std::vector<std::int32_t> sspec;
  long sacc = 0, srounds = 0;
  ASSERT_TRUE(m->mtp_decode(ce, f2, Ns, sspec, 1, &sacc, &srounds, sc));
  std::printf("[qwen4_exp] MTP decode, sampled (temperature 0.7, top-k 20, "
              "top-p 0.8, presence 1.5): %zu tokens in %ld rounds, %ld "
              "drafts kept; %s the pipelined sampler's\n", sspec.size(),
              srounds, sacc, sspec == ps ? "the same tokens as" :
                                           "NOT the same tokens as");
  EXPECT_TRUE(sspec == ps);
}

// The DFlash 2 drafter (incoai/Qwen3.8-Flash-Next-DFlash2) speculating for
// the model: a math word problem through the chat template, decoded plain
// and speculatively -- greedy at the drafter's trained block, at block 4
// (the reference's serving setting) and adaptive, then sampled with the
// chat pipelines' penalties -- every speculative run the plain decode's
// tokens EXACTLY, and the drafter drafting (well over the one token a
// round a decode without drafts keeps). Its taps are the
// reference's capture (sglang's patched qwen4_exp): layers 5 / 15 / 25 /
// 35 / 45's completed outputs averaged over the four hyper-connection
// streams. Acceptance cannot check that definition -- MEASURED, every tap
// a layer early or late keeps as many drafts -- so it is the residual
// that is verified (prefill_matches_reference) and the capture that is
// read off the reference. VPIPE_QWEN4_EXP_TEST_MODEL_PATH +
// VPIPE_QWEN4_EXP_DFLASH; VPIPE_QWEN4_EXP_DFLASH_TOKENS (default 96),
// VPIPE_QWEN4_EXP_DFLASH_BITS (the drafter's holding, default w8).
TEST(qwen4_exp, dflash_decode_keeps_the_plain_tokens)
{
  const char* mp = std::getenv("VPIPE_QWEN4_EXP_TEST_MODEL_PATH");
  const char* dp = std::getenv("VPIPE_QWEN4_EXP_DFLASH");
  if (mp == nullptr || dp == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  using vpipe::genai::MetalQwenModel;
  vpipe::genai::ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(mp);
  ASSERT_TRUE(mcfg.has_value());
  if (!mcfg.has_value()) { return; }
  MetalQwenModel::Config c = MetalQwenModel::config_from(*mcfg);
  c.use_bf16 = std::getenv("VPIPE_QWEN4_EXP_F16") == nullptr;
  c.page_tokens = 256;
  c.max_pages = 64;
  c.max_seq = 8192;
  std::size_t cache_mb = 36864;
  if (const char* e = std::getenv("VPIPE_QWEN4_EXP_CACHE_MB")) {
    cache_mb = (std::size_t)std::atoll(e);
  }
  c.expert_cache_bytes = cache_mb << 20;
  auto tok = vpipe::genai::Tokenizer::from_huggingface_json(
      std::string(mp) + "/tokenizer.json", &sess);
  ASSERT_TRUE(tok != nullptr);
  if (tok == nullptr) { return; }
  auto tpl = vpipe::genai::make_chat_template(*mcfg, *tok,
                                              /*disable_thinking=*/true);
  ASSERT_TRUE(tpl != nullptr);
  if (tpl == nullptr) { return; }
  std::vector<std::int32_t> prompt;
  // VPIPE_QWEN4_EXP_DFLASH_PROMPT: another prompt (a story runs longer
  // than the arithmetic before its first near-tie).
  const char* dpr = std::getenv("VPIPE_QWEN4_EXP_DFLASH_PROMPT");
  tpl->render_user_turn(
      dpr != nullptr ? dpr :
      "Natalia sold clips to 48 of her friends in April, and then she sold "
      "half as many clips in May. How many clips did Natalia sell "
      "altogether in April and May? Explain your reasoning step by step.",
      /*is_first_turn=*/true, &prompt);
  int N = 96;
  if (const char* e = std::getenv("VPIPE_QWEN4_EXP_DFLASH_TOKENS")) {
    N = std::max(8, std::atoi(e));
  }
  auto m = MetalQwenModel::load(mp, mc, c);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  vpipe::genai::MetalDFlashDrafter::Options opt;
  opt.target_bf16 = c.use_bf16;
  // VPIPE_QWEN4_EXP_DFLASH_BITS: the drafter held dense (0) / w8 / w4.
  if (const char* e = std::getenv("VPIPE_QWEN4_EXP_DFLASH_BITS")) {
    opt.quant_bits = std::atoi(e);
  }
  std::string err;
  auto d = vpipe::genai::MetalDFlashDrafter::load(
      vpipe::genai::WeightSet::open(dp, &sess), mc, opt, &err);
  ASSERT_TRUE(d != nullptr);
  if (d == nullptr) {
    std::printf("[qwen4_exp] drafter: %s\n", err.c_str());
    return;
  }
  const int trained = d->config().block_size;
  // The plain decode with no drafter attached: attaching one must change
  // nothing of it (its prefill taps the drafter's layers -- dispatches of
  // its own, which once moved an encoder split onto an op whose bindings
  // did not survive it, and the plain decode departed 110 tokens in).
  std::vector<std::int32_t> pre_plain;
  {
    auto am = [](const std::vector<float>& v) {
      std::size_t a = 0;
      for (std::size_t i = 1; i < v.size(); ++i) {
        if (v[i] > v[a]) { a = i; }
      }
      return (std::int32_t)a;
    };
    const auto cx = m->context_manager()->acquire_root();
    const std::int32_t first = am(m->prefill(cx, prompt));
    pre_plain.push_back(first);
    vpipe::genai::GpuSamplerParams g0;
    if (m->pdecode_begin(cx, first, prompt, g0, N)) {
      for (int i = 1; i < N; ++i) {
        if (!m->pdecode_commit(cx)) { break; }
        const std::int32_t t = m->pdecode_next(cx);
        if (t < 0) { break; }
        pre_plain.push_back(t);
      }
      m->pdecode_end(cx);
    }
    m->context_manager()->release(cx);
  }
  ASSERT_TRUE(m->attach_dflash(std::move(d), &err));
  if (!m->has_dflash()) {
    std::printf("[qwen4_exp] attach: %s\n", err.c_str());
    return;
  }
  auto argmax = [](const std::vector<float>& v) {
    std::size_t a = 0;
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > v[a]) { a = i; }
    }
    return (std::int32_t)a;
  };
  using clk = std::chrono::steady_clock;
  // The plain (pipelined) decode of N tokens on a fresh context.
  auto run_plain = [&](const vpipe::genai::GpuSamplerParams& sp, int n,
                       double* ms) {
    const auto cx = m->context_manager()->acquire_root();
    const std::int32_t first = argmax(m->prefill(cx, prompt));
    std::vector<std::int32_t> out{first};
    const auto t0 = clk::now();
    if (m->pdecode_begin(cx, first, prompt, sp, n)) {
      for (int i = 1; i < n; ++i) {
        if (!m->pdecode_commit(cx)) { break; }
        const std::int32_t t = m->pdecode_next(cx);
        if (t < 0) { break; }
        out.push_back(t);
      }
      m->pdecode_end(cx);
    }
    if (ms != nullptr) {
      *ms = std::chrono::duration<double, std::milli>(clk::now() - t0)
                .count();
    }
    m->context_manager()->release(cx);
    return out;
  };
  // The speculative decode of n tokens at `block` (0: adaptive).
  auto run_spec = [&](const vpipe::genai::GpuSamplerParams& sp, int n,
                      int block, MetalQwenModel::SpecStats* ss, double* ms) {
    const auto cx = m->context_manager()->acquire_root();
    const std::int32_t first = argmax(m->prefill(cx, prompt));
    vpipe::genai::MtpDecodeCtl ctl;
    ctl.sampler = sp;
    ctl.prime = prompt;
    std::vector<std::int32_t> out;
    const auto t0 = clk::now();
    const bool ok = m->dflash_decode(cx, first, n, out, ctl, block, ss);
    *ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    m->context_manager()->release(cx);
    return ok ? out : std::vector<std::int32_t>{};
  };
  vpipe::genai::GpuSamplerParams gs;
  double cold_ms = 0.0, warm_ms = 0.0;
  const std::vector<std::int32_t> plain = run_plain(gs, N, &cold_ms);
  ASSERT_TRUE((int)plain.size() == N);
  {
    int same = 0;
    while (same < (int)std::min(plain.size(), pre_plain.size()) &&
           plain[(std::size_t)same] == pre_plain[(std::size_t)same]) {
      ++same;
    }
    std::printf("[qwen4_exp] the plain decode with no drafter attached: "
                "the first %d of %zu the same\n", same, plain.size());
    EXPECT_TRUE(pre_plain == plain);
  }
  std::printf("[qwen4_exp] DFlash 2: a %zu-token prompt, %d tokens: %s\n",
              prompt.size(), N, tok->decode(plain).c_str());
  struct Arm {
    const char* name;
    int block;
  };
  const Arm arms[] = {{"trained", trained}, {"block 4", 4},
                      {"adaptive", 0}};
  for (const Arm& a : arms) {
    MetalQwenModel::SpecStats ss;
    double ms = 0.0;
    const std::vector<std::int32_t> spec = run_spec(gs, N, a.block, &ss, &ms);
    int same = 0;
    while (same < (int)std::min(spec.size(), plain.size()) &&
           spec[(std::size_t)same] == plain[(std::size_t)same]) {
      ++same;
    }
    const double per_round =
        ss.rounds > 0 ? (double)spec.size() / (double)ss.rounds : 0.0;
    std::printf("[qwen4_exp] DFlash 2 greedy, %s (%d): %zu tokens in %ld "
                "rounds (%.2f a round), %ld of %ld drafts kept, %.0f ms a "
                "token; the first %d of %zu the plain decode's\n", a.name,
                a.block, spec.size(), ss.rounds, per_round, ss.accepted,
                ss.drafted, ms / std::max<std::size_t>(spec.size(), 1),
                same, plain.size());
    EXPECT_TRUE(spec.size() == plain.size() && same == (int)plain.size());
    // Engaged: a decode without drafts keeps one a round.
    if (a.block != 0) { EXPECT_TRUE(per_round > 1.8); }
  }
  (void)run_plain(gs, N, &warm_ms);
  std::printf("[qwen4_exp] the plain decode: %.0f ms a token warm (%.0f "
              "cold)\n", warm_ms / (double)N, cold_ms / (double)N);
  // SAMPLED, with the chat pipelines' penalties: each verify row samples
  // with the seed and seen-set a decode step would hold there.
  vpipe::genai::GpuSamplerParams ss;
  ss.greedy = false;
  ss.temperature = 0.7f;
  ss.top_k = 20;
  ss.top_p = 0.8f;
  ss.presence_penalty = 1.5f;
  ss.seed = 42;
  const int Ns = std::min(N, 64);
  const std::vector<std::int32_t> ps = run_plain(ss, Ns, nullptr);
  MetalQwenModel::SpecStats sst;
  double sms = 0.0;
  const std::vector<std::int32_t> sspec = run_spec(ss, Ns, 4, &sst, &sms);
  std::printf("[qwen4_exp] DFlash 2 sampled (temperature 0.7, top-k 20, "
              "top-p 0.8, presence 1.5), block 4: %zu tokens in %ld rounds, "
              "%ld drafts kept; %s the pipelined sampler's\n", sspec.size(),
              sst.rounds, sst.accepted,
              sspec == ps ? "the same tokens as" : "NOT the same tokens as");
  EXPECT_TRUE(sspec == ps);
}

// What makes the speculative decode keep the plain one's tokens EXACTLY: a
// few rows forwarded together (a verify) compute each row as a one-row
// decode step does -- the same logits, bit for bit, at 2, 3 and 5 rows
// (the batched GEMV, the MoE block's router and shared expert a row at a
// time; every other kernel is per row already).
TEST(qwen4_exp, verify_rows_are_decode_rows)
{
  const char* mp = std::getenv("VPIPE_QWEN4_EXP_TEST_MODEL_PATH");
  const char* gp = std::getenv("VPIPE_QWEN4_EXP_GOLDEN");
  if (mp == nullptr || gp == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  using vpipe::genai::MetalQwenModel;
  vpipe::genai::ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(mp);
  ASSERT_TRUE(mcfg.has_value());
  if (!mcfg.has_value()) { return; }
  MetalQwenModel::Config c = MetalQwenModel::config_from(*mcfg);
  c.use_bf16 = std::getenv("VPIPE_QWEN4_EXP_F16") == nullptr;
  c.page_tokens = 256;
  c.max_pages = 64;
  c.max_seq = 8192;
  std::size_t cache_mb = 36864;
  if (const char* e = std::getenv("VPIPE_QWEN4_EXP_CACHE_MB")) {
    cache_mb = (std::size_t)std::atoll(e);
  }
  c.expert_cache_bytes = cache_mb << 20;
  const auto all = read_raw_<std::int32_t>(std::string(gp) + "/ids.i32");
  if (all.size() < 32) { return; }
  auto m = MetalQwenModel::load(mp, mc, c);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  const std::size_t P = all.size() - 12;
  const std::vector<std::int32_t> prompt(all.begin(), all.begin() + (long)P);
  for (const int rows : {2, 3, 5}) {
    const std::vector<std::int32_t> next(all.begin() + (long)P,
                                         all.begin() + (long)P + rows);
    const auto ca = m->context_manager()->acquire_root();
    const auto cb = m->context_manager()->acquire_root();
    ASSERT_TRUE(!m->prefill(ca, prompt).empty());
    ASSERT_TRUE(!m->prefill(cb, prompt).empty());
    std::vector<float> steps;
    for (const std::int32_t t : next) {
      const std::vector<float> lg = m->forward(ca, t);
      steps.insert(steps.end(), lg.begin(), lg.end());
    }
    const std::vector<float> batch = m->prefill_all_logits(cb, next);
    ASSERT_TRUE(batch.size() == steps.size() && !steps.empty());
    std::size_t diff = 0;
    for (std::size_t i = 0; i < steps.size() && i < batch.size(); ++i) {
      diff += steps[i] != batch[i];
    }
    std::printf("[qwen4_exp] %d rows together against %d decode steps: %zu "
                "of %zu logits differ\n", rows, rows, diff, steps.size());
    EXPECT_TRUE(diff == 0);
    m->context_manager()->release(ca);
    m->context_manager()->release(cb);
  }
}

// A SPECULATIVE DECODE LEAVES THE PLAIN DECODE'S STATE. A round that keeps
// fewer rows than it verified replays the kept ones into the GDN / PLE
// state, and that state has to be the one a decode step a token leaves --
// to the bit, or the tokens part at the first near-tie however long they
// agreed. A token run sees a wrong state only at such a tie; this reads
// the state itself: after the draft head's decode (and DFlash 2's, with
// VPIPE_QWEN4_EXP_DFLASH), the next token's logits on that context against
// a context that took the same tokens a decode step each. Rounds that
// rejected a draft are required, or no replay ran (128 tokens: ~10
// replays an arm). MUTATION: the replay's q / k norm eps x4 moves 237,966
// of 248,320 logits.
TEST(qwen4_exp, speculative_decodes_leave_the_plain_state)
{
  const char* mp = std::getenv("VPIPE_QWEN4_EXP_TEST_MODEL_PATH");
  if (mp == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  using vpipe::genai::MetalQwenModel;
  vpipe::genai::ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(mp);
  ASSERT_TRUE(mcfg.has_value());
  if (!mcfg.has_value()) { return; }
  MetalQwenModel::Config c = MetalQwenModel::config_from(*mcfg);
  c.use_bf16 = std::getenv("VPIPE_QWEN4_EXP_F16") == nullptr;
  c.page_tokens = 256;
  c.max_pages = 64;
  c.max_seq = 8192;
  std::size_t cache_mb = 36864;
  if (const char* e = std::getenv("VPIPE_QWEN4_EXP_CACHE_MB")) {
    cache_mb = (std::size_t)std::atoll(e);
  }
  c.expert_cache_bytes = cache_mb << 20;
  auto tok = vpipe::genai::Tokenizer::from_huggingface_json(
      std::string(mp) + "/tokenizer.json", &sess);
  ASSERT_TRUE(tok != nullptr);
  if (tok == nullptr) { return; }
  auto tpl = vpipe::genai::make_chat_template(*mcfg, *tok,
                                              /*disable_thinking=*/true);
  ASSERT_TRUE(tpl != nullptr);
  if (tpl == nullptr) { return; }
  std::vector<std::int32_t> prompt;
  tpl->render_user_turn(
      "Natalia sold clips to 48 of her friends in April, and then she sold "
      "half as many clips in May. How many clips did Natalia sell "
      "altogether in April and May? Explain your reasoning step by step.",
      /*is_first_turn=*/true, &prompt);
  auto m = MetalQwenModel::load(mp, mc, c);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  auto argmax = [](const std::vector<float>& v) {
    std::size_t a = 0;
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > v[a]) { a = i; }
    }
    return (std::int32_t)a;
  };
  constexpr int N = 128;
  // `spec` decodes N tokens on a prefilled context from its first token,
  // reporting the drafts it offered and the ones it kept.
  auto check = [&](const char* name,
                   const std::function<bool(vpipe::genai::ContextId,
                                            std::int32_t,
                                            std::vector<std::int32_t>&,
                                            long*, long*)>& spec) {
    auto* cm = m->context_manager();
    const auto cb = cm->acquire_root();
    const std::int32_t first = argmax(m->prefill(cb, prompt));
    std::vector<std::int32_t> out;
    long drafted = 0, kept = 0;
    const bool ok = spec(cb, first, out, &drafted, &kept);
    // What the context took in: the prompt and the first `took` of `out`
    // (out_ids starts with the first token). A decode that ends on a pick
    // has ingested every token it returned; then the next is whatever the
    // plain run picks after them.
    const int took = cm->seq_len_of(cb) - (int)prompt.size();
    ASSERT_TRUE(ok && !out.empty() && out[0] == first && took > 0 &&
                took <= (int)out.size());
    if (!ok || out.empty() || took <= 0 || took > (int)out.size()) {
      cm->release(cb);
      return;
    }
    const auto ca = cm->acquire_root();
    std::vector<float> lp = m->prefill(ca, prompt);
    for (int i = 0; i < took; ++i) {
      lp = m->forward(ca, out[(std::size_t)i]);
    }
    const std::int32_t nxt =
        took < (int)out.size() ? out[(std::size_t)took] : argmax(lp);
    const std::vector<float> la = m->forward(ca, nxt);
    const std::vector<float> lb = m->forward(cb, nxt);
    std::size_t diff = 0;
    for (std::size_t i = 0; i < la.size() && i < lb.size(); ++i) {
      diff += la[i] != lb[i];
    }
    std::printf("[qwen4_exp] %s: %zu tokens, %ld of %ld drafts kept; the "
                "next token's logits after it against %d decode steps: %zu "
                "of %zu differ\n", name, out.size(), kept, drafted, took,
                diff, la.size());
    EXPECT_TRUE(drafted > kept);           // a rejected draft: a replay ran
    EXPECT_TRUE(!la.empty() && la.size() == lb.size() && diff == 0);
    cm->release(ca);
    cm->release(cb);
  };
  check("draft head", [&](vpipe::genai::ContextId cx, std::int32_t first,
                          std::vector<std::int32_t>& out, long* drafted,
                          long* kept) {
    vpipe::genai::MtpDecodeCtl ctl;
    ctl.drafted = drafted;
    long rounds = 0;
    return m->mtp_decode(cx, first, N, out, 1, kept, &rounds, ctl);
  });
  const char* dp = std::getenv("VPIPE_QWEN4_EXP_DFLASH");
  if (dp == nullptr) { return; }
  vpipe::genai::MetalDFlashDrafter::Options opt;
  opt.target_bf16 = c.use_bf16;
  std::string err;
  auto d = vpipe::genai::MetalDFlashDrafter::load(
      vpipe::genai::WeightSet::open(dp, &sess), mc, opt, &err);
  ASSERT_TRUE(d != nullptr);
  if (d == nullptr) { return; }
  ASSERT_TRUE(m->attach_dflash(std::move(d), &err));
  if (!m->has_dflash()) { return; }
  check("DFlash 2, block 4", [&](vpipe::genai::ContextId cx,
                                 std::int32_t first,
                                 std::vector<std::int32_t>& out,
                                 long* drafted, long* kept) {
    MetalQwenModel::SpecStats st;
    const bool ok = m->dflash_decode(cx, first, N, out, {}, 4, &st);
    *drafted = st.drafted;
    *kept = st.accepted;
    return ok;
  });
}

// PAST THE QSA THRESHOLD: a ~2400-token prompt (VPIPE_QWEN4_EXP_LONG_GOLDEN,
// the dumper's --prompt-file run, its last 48 rows kept), so each query
// past 2051 visible tokens attends only the indexer's top 512 blocks of 4
// and its tail. The last rows' logits against the fp32 reference; with
// VPIPE_QWEN_QSA_OFF set the same run is dense -- the A/B that says the
// sparse rows are the reason.
TEST(qwen4_exp, long_context_matches_reference)
{
  const char* mp = std::getenv("VPIPE_QWEN4_EXP_TEST_MODEL_PATH");
  const char* gp = std::getenv("VPIPE_QWEN4_EXP_LONG_GOLDEN");
  if (mp == nullptr || gp == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  using vpipe::genai::MetalQwenModel;
  vpipe::genai::ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(mp);
  ASSERT_TRUE(mcfg.has_value());
  if (!mcfg.has_value()) { return; }
  MetalQwenModel::Config c = MetalQwenModel::config_from(*mcfg);
  c.use_bf16 = std::getenv("VPIPE_QWEN4_EXP_F16") == nullptr;
  c.page_tokens = 256;
  c.max_pages = 64;
  c.max_seq = 8192;
  std::size_t cache_mb = 36864;
  if (const char* e = std::getenv("VPIPE_QWEN4_EXP_CACHE_MB")) {
    cache_mb = (std::size_t)std::atoll(e);
  }
  c.expert_cache_bytes = cache_mb << 20;
  const std::string g = gp;
  const auto ids = read_raw_<std::int32_t>(g + "/ids.i32");
  const auto ref = read_raw_<float>(g + "/logits_f32.f32");
  const std::size_t T = ids.size();
  constexpr std::size_t kRows = 48;
  ASSERT_TRUE(T > 2100 && ref.size() % kRows == 0);
  if (T <= 2100 || ref.size() % kRows != 0) { return; }
  const std::size_t V = ref.size() / kRows;
  auto m = MetalQwenModel::load(mp, mc, c);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  // The sparse attention's picks for the last row, every full layer:
  // the logits alone cannot tell QSA from dense at this length.
  const std::string sel_path =
      std::string(std::getenv("TMPDIR") != nullptr ? std::getenv("TMPDIR")
                                                   : "/tmp") +
      "/qwen4_exp_qsa_sel.i32";
  std::remove(sel_path.c_str());
  setenv("VPIPE_QWEN_QSA_DUMP", sel_path.c_str(), 1);
  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<float> got = m->prefill_all_logits(
      m->root_context(), std::vector<std::int32_t>(ids.begin(), ids.end()));
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
  unsetenv("VPIPE_QWEN_QSA_DUMP");
  ASSERT_TRUE(got.size() == T * V);
  if (got.size() != T * V) { return; }
  const bool off = std::getenv("VPIPE_QWEN_QSA_OFF") != nullptr;
  if (!off) {
    // Tokens picked by one run and not the other, per full layer, against
    // the reference's fp32 -- and its own bf16's count as the bar.
    const auto sel = read_raw_<std::int32_t>(sel_path);
    const auto q32 = read_raw_<std::int32_t>(g + "/qsa_f32.i32");
    const auto q16 = read_raw_<std::int32_t>(g + "/qsa_bf16.i32");
    constexpr int kFull = 12;
    const std::size_t W = q32.size() / kFull;
    ASSERT_TRUE(q32.size() == kFull * W && sel.size() >= kFull * (W + 1));
    if (q32.size() != kFull * W || sel.size() < kFull * (W + 1)) { return; }
    auto picks = [&](const std::int32_t* p, std::size_t w) {
      std::vector<char> on(T, 0);
      for (std::size_t i = 0; i < w; ++i) {
        if (p[i] >= 0 && (std::size_t)p[i] < T) { on[(std::size_t)p[i]] = 1; }
      }
      return on;
    };
    auto differ = [&](const std::vector<char>& a,
                      const std::vector<char>& b) {
      int d = 0;
      for (std::size_t t = 0; t < T; ++t) { d += a[t] != b[t]; }
      return d / 2;
    };
    int ours_total = 0, ref_total = 0;
    for (int f = 0; f < kFull; ++f) {
      const std::int32_t cnt = sel[(std::size_t)f * (W + 1) + W];
      ASSERT_TRUE(cnt > 0 && (std::size_t)cnt <= W);
      if (cnt <= 0 || (std::size_t)cnt > W) { return; }
      const auto a = picks(sel.data() + (std::size_t)f * (W + 1),
                           (std::size_t)cnt);
      const auto r = picks(q32.data() + (std::size_t)f * W, W);
      const int d = differ(a, r);
      int d16 = -1;
      if (q16.size() == q32.size()) {
        d16 = differ(picks(q16.data() + (std::size_t)f * W, W), r);
        ref_total += d16;
      }
      ours_total += d;
      std::printf("[qwen4_exp]   full layer %2d: %d tokens picked, %d off "
                  "the reference's (its bf16 %d)\n", f, cnt, d, d16);
    }
    std::printf("[qwen4_exp]   QSA picks off the fp32 reference's: %d "
                "tokens (its bf16 %d)\n", ours_total, ref_total);
    if (q16.size() == q32.size()) {
      EXPECT_TRUE(ours_total <= 2 * ref_total);
    }
  }
  const float* last = got.data() + (T - kRows) * V;
  // The reference's own bf16 run is the bar: at 2422 tokens a 48-layer
  // bf16 stack is far from fp32 on its own (0.1168, 40 of 42 decisive).
  const auto ref16 = read_raw_<float>(g + "/logits_bf16.f32");
  const bool has16 = ref16.size() == ref.size();
  auto argmax = [&](const float* r) {
    std::size_t a = 0;
    for (std::size_t v = 1; v < V; ++v) {
      if (r[v] > r[a]) { a = v; }
    }
    return a;
  };
  int decisive = 0, agree = 0, agree16 = 0;
  for (std::size_t t = 0; t < kRows; ++t) {
    const float* r = ref.data() + t * V;
    const std::size_t a = argmax(r);
    std::size_t b = a == 0 ? 1 : 0;
    for (std::size_t v = 0; v < V; ++v) {
      if (v != a && r[v] > r[b]) { b = v; }
    }
    const float margin = r[a] - r[b];
    if (margin <= 0.5f) { continue; }
    ++decisive;
    const bool ours = argmax(last + t * V) == a;
    agree += ours;
    if (has16) { agree16 += argmax(ref16.data() + t * V) == a; }
    if (!ours) {
      std::printf("[qwen4_exp]   row %zu missed (fp32 margin %.3f)\n",
                  t, margin);
    }
  }
  const double e = rel_l2_(ref.data(), last, kRows * V);
  const double e16 =
      has16 ? rel_l2_(ref.data(), ref16.data(), kRows * V) : 0.1168;
  std::printf("[qwen4_exp] %zu-token prefill (%s, QSA %s) in %.0f ms: the "
              "last %zu rows' logits %.4f from fp32 (reference bf16 "
              "%.4f); decisive argmax %d/%d (reference bf16 %d)\n",
              T, c.use_bf16 ? "bf16" : "f16", off ? "OFF" : "on", ms, kRows,
              e, e16, agree, decisive, has16 ? agree16 : 40);
  if (!off) {
    EXPECT_TRUE(e < 1.15 * e16);
    EXPECT_TRUE(agree + 3 >= (has16 ? agree16 : 40));
  }
}

// A PREFILL IN PIECES: the long golden's prompt but its last 48 rows
// through prefill() -- one piece at hc_piece_rows (8192 on a 64 GB box),
// and again in four calls on another context, pieces of ~600 rows each
// run against the K/V, GDN state and PLE history the ones before left --
// then those 48 rows by one forward continuing the context, their logits
// against the fp32 reference's as long_context_matches_reference holds
// its one forward (its bf16 x 1.15, the decisive picks); and the tail
// decoded a row at a time. The tail crosses QSA's line (2051 visible
// tokens). VPIPE_QWEN4_EXP_PIECE_BENCH=<tokens> first times a prefill of
// that many synthetic tokens on the cold cache (VPIPE_QWEN4_EXP_PIECE_SEED
// other tokens; VPIPE_QWEN_PREFILL_CHUNK=2048 the old pieces' arm).
TEST(qwen4_exp, prefill_pieces_agree)
{
  const char* mp = std::getenv("VPIPE_QWEN4_EXP_TEST_MODEL_PATH");
  const char* gp = std::getenv("VPIPE_QWEN4_EXP_LONG_GOLDEN");
  if (mp == nullptr || gp == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  using vpipe::genai::MetalQwenModel;
  vpipe::genai::ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(mp);
  ASSERT_TRUE(mcfg.has_value());
  if (!mcfg.has_value()) { return; }
  MetalQwenModel::Config c = MetalQwenModel::config_from(*mcfg);
  c.use_bf16 = std::getenv("VPIPE_QWEN4_EXP_F16") == nullptr;
  c.page_tokens = 256;
  c.max_pages = 72;
  c.max_seq = 16384;
  std::size_t cache_mb = 36864;
  if (const char* e = std::getenv("VPIPE_QWEN4_EXP_CACHE_MB")) {
    cache_mb = (std::size_t)std::atoll(e);
  }
  c.expert_cache_bytes = cache_mb << 20;
  const std::string g = gp;
  const auto ids32 = read_raw_<std::int32_t>(g + "/ids.i32");
  const auto ref = read_raw_<float>(g + "/logits_f32.f32");
  const auto ref16 = read_raw_<float>(g + "/logits_bf16.f32");
  constexpr std::size_t kRows = 48;
  const std::size_t T = ids32.size();
  ASSERT_TRUE(T > 2100 && ref.size() % kRows == 0 &&
              ref16.size() == ref.size());
  if (T <= 2100 || ref.size() % kRows != 0 || ref16.size() != ref.size()) {
    return;
  }
  const std::size_t V = ref.size() / kRows;
  const std::vector<std::int32_t> ids(ids32.begin(), ids32.end());
  auto m = MetalQwenModel::load(mp, mc, c);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  // The bench first, on the cold cache.
  if (const char* b = std::getenv("VPIPE_QWEN4_EXP_PIECE_BENCH")) {
    const int n = std::max(1, std::atoi(b));
    // VPIPE_QWEN4_EXP_PIECE_SEED: other tokens (n-gram rows no earlier run
    // left in the page cache)
    const char* sd = std::getenv("VPIPE_QWEN4_EXP_PIECE_SEED");
    const long seed = sd != nullptr ? std::atol(sd) : 0;
    std::vector<std::int32_t> syn((std::size_t)n);
    for (int i = 0; i < n; ++i) {
      syn[(std::size_t)i] =
          1000 + (std::int32_t)((i * 7919L + seed * 104729L) % 50000);
    }
    const auto cx = m->context_manager()->acquire_root();
    const auto t0 = std::chrono::steady_clock::now();
    const auto lg = m->prefill(cx, syn);
    const double sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("[qwen4_exp] piece bench: %d tokens prefilled in %.1f s "
                "(%.0f tok/s)\n", n, sec, n / sec);
    EXPECT_TRUE(!lg.empty());
    m->forget_context(cx);
    m->context_manager()->release(cx);
  }
  // The prompt but its last 48 rows in pieces, then those rows' logits by
  // one forward continuing the context.
  const std::size_t P = T - kRows;
  const std::vector<std::int32_t> tail(ids.begin() + P, ids.end());
  auto run = [&](std::size_t calls) {
    const auto cx = m->context_manager()->acquire_root();
    const std::size_t step = (P + calls - 1) / calls;
    for (std::size_t i = 0; i < P; i += step) {
      const std::size_t e = std::min(P, i + step);
      if (m->prefill(cx, std::vector<std::int32_t>(ids.begin() + i,
                                                   ids.begin() + e))
              .empty()) {
        return std::vector<float>{};
      }
    }
    std::vector<float> lg = m->prefill_all_logits(cx, tail);
    m->forget_context(cx);
    m->context_manager()->release(cx);
    return lg;
  };
  const std::vector<float> whole = run(1), parts = run(4);
  ASSERT_TRUE(whole.size() == ref.size() && parts.size() == ref.size());
  if (whole.size() != ref.size() || parts.size() != ref.size()) { return; }
  // ...and the tail DECODED a row at a time after one piece: past 2051
  // tokens each step's QSA select runs (in two phases with
  // VPIPE_QWEN_QSA_TWO=2 -- the same blocks, so the same logits: compare
  // the digest printed against a run with it 0).
  std::vector<float> dec;
  {
    const auto cx = m->context_manager()->acquire_root();
    ASSERT_TRUE(!m->prefill(cx, std::vector<std::int32_t>(
                                    ids.begin(), ids.begin() + P)).empty());
    for (const std::int32_t t : tail) {
      const std::vector<float> lg = m->forward(cx, t);
      ASSERT_TRUE(lg.size() == V);
      if (lg.size() != V) { return; }
      dec.insert(dec.end(), lg.begin(), lg.end());
    }
    m->forget_context(cx);
    m->context_manager()->release(cx);
  }
  std::uint64_t digest = 1469598103934665603ull;
  for (const float f : dec) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    digest = (digest ^ u) * 1099511628211ull;
  }
  const double ed = rel_l2_(ref.data(), dec.data(), ref.size());
  std::printf("[qwen4_exp] the same %zu rows decoded a row at a time: %.4f "
              "from fp32 (logits digest %016llx; %ld two-phase QSA "
              "selects)\n", kRows, ed, (unsigned long long)digest,
              m->qsa_two_phase_selects());
  EXPECT_TRUE(ed < 1.15 * rel_l2_(ref.data(), ref16.data(), ref.size()));
  auto argmax = [&](const float* x) {
    std::size_t a = 0;
    for (std::size_t v = 1; v < V; ++v) {
      if (x[v] > x[a]) { a = v; }
    }
    return a;
  };
  // the fp32-decisive rows (margin > 0.5) each run and the reference's
  // bf16 pick as fp32 does
  int decisive = 0, aw = 0, ap = 0, a16 = 0;
  for (std::size_t t = 0; t < kRows; ++t) {
    const float* r = ref.data() + t * V;
    const std::size_t a = argmax(r);
    std::size_t b = a == 0 ? 1 : 0;
    for (std::size_t v = 0; v < V; ++v) {
      if (v != a && r[v] > r[b]) { b = v; }
    }
    if (r[a] - r[b] <= 0.5f) { continue; }
    ++decisive;
    aw += argmax(whole.data() + t * V) == a;
    ap += argmax(parts.data() + t * V) == a;
    a16 += argmax(ref16.data() + t * V) == a;
  }
  const double bar = rel_l2_(ref.data(), ref16.data(), ref.size());
  const double ew = rel_l2_(ref.data(), whole.data(), ref.size());
  const double ep = rel_l2_(ref.data(), parts.data(), ref.size());
  std::printf("[qwen4_exp] a %zu-token prompt in pieces of up to %d rows, "
              "its last %zu rows' logits %.4f from fp32, after four calls "
              "%.4f (reference bf16 %.4f); decisive argmax %d / %d of %d "
              "(reference bf16 %d)\n", P, m->hc_piece_rows(c), kRows, ew, ep,
              bar, aw, ap, decisive, a16);
  EXPECT_TRUE(ew < 1.15 * bar);
  EXPECT_TRUE(ep < 1.15 * bar);
  EXPECT_TRUE(aw + 3 >= a16 && ap + 3 >= a16);
}

// AN IMAGE TURN (VPIPE_QWEN4_EXP_IMAGE_GOLDEN: one or more of the dumper's
// --image runs, ':'-separated, the same 512^2 picture: a short turn with
// every row kept, and the picture before the long golden's ~2400-token
// prompt with its last 48 rows). The vision tower's rows against the
// reference's; then the language model fed the REFERENCE's rows, so the
// tower's error stays out of it -- the ids with their image placeholders
// for the n-gram PLE (the short turn's rows right after the picture are
// where a wrong id shows: hashed as 0 they move the long turn's last rows
// only within its bar), the 3-axis mROPE positions, and past 2051 tokens
// the QSA indexer ranking the image's blocks at the image's positions:
// the kept rows' logits and the last row's picks, against the fp32
// reference and its own bf16 run (whose bar also carries its bf16 tower).
TEST(qwen4_exp, image_turn_matches_reference)
{
  const char* mp = std::getenv("VPIPE_QWEN4_EXP_TEST_MODEL_PATH");
  const char* gl = std::getenv("VPIPE_QWEN4_EXP_IMAGE_GOLDEN");
  if (mp == nullptr || gl == nullptr) { return; }
  std::vector<std::string> goldens;
  for (std::string rest = gl; !rest.empty();) {
    const std::size_t c = rest.find(':');
    goldens.push_back(rest.substr(0, c));
    rest = c == std::string::npos ? "" : rest.substr(c + 1);
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  using vpipe::genai::MetalQwenModel;
  using vpipe::genai::MetalQwenVisionEncoder;
  vpipe::genai::ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(mp);
  ASSERT_TRUE(mcfg.has_value());
  if (!mcfg.has_value()) { return; }
  const int H = mcfg->hidden;
  const std::size_t V = (std::size_t)mcfg->vocab_size;
  constexpr int kSide = 512;
  constexpr std::int32_t kImagePad = 248056;

  // The tower, on the goldens' own pixels (a 512^2 picture neither
  // processor resizes).
  {
    const std::string& g = goldens.front();
    const auto rgb = read_raw_<std::uint8_t>(g + "/image_rgb.u8");
    const auto img32 = read_raw_<float>(g + "/image_embeds_f32.f32");
    const auto img16 = read_raw_<float>(g + "/image_embeds_bf16.f32");
    ASSERT_TRUE(rgb.size() == (std::size_t)3 * kSide * kSide &&
                !img32.empty());
    if (rgb.size() != (std::size_t)3 * kSide * kSide || img32.empty()) {
      return;
    }
    auto vis = MetalQwenVisionEncoder::load(
        mp, mc, MetalQwenVisionEncoder::config_from(*mcfg));
    ASSERT_TRUE(vis != nullptr);
    if (vis == nullptr) { return; }
    const auto res = vis->encode(rgb.data(), kSide, kSide);
    const std::size_t n_img = img32.size() / (std::size_t)H;
    ASSERT_TRUE((std::size_t)res.n_tokens == n_img);
    if ((std::size_t)res.n_tokens != n_img) { return; }
    const auto* e =
        static_cast<const std::uint16_t*>(res.embeddings.contents());
    std::vector<float> ours(n_img * (std::size_t)H);
    for (std::size_t i = 0; i < ours.size(); ++i) {
      ours[i] = (float)(res.is_bf16 ? from_bf16_(e[i]) : from_f16_(e[i]));
    }
    const double et = rel_l2_(img32.data(), ours.data(), ours.size());
    const double bt = img16.size() == img32.size()
                          ? rel_l2_(img32.data(), img16.data(), img32.size())
                          : 0.0;
    std::printf("[qwen4_exp] vision tower (%s), %zu rows: %.4f from fp32 "
                "(the reference's bf16 %.4f)\n", res.is_bf16 ? "bf16" : "f16",
                n_img, et, bt);
    EXPECT_TRUE(et < std::max(0.02, 1.5 * bt));
  }

  // Both precisions -- bf16 the model's own, f16 the closer to fp32 and
  // the tight check (VPIPE_QWEN4_EXP_F16=1: f16 alone).
  std::vector<bool> precisions = {true, false};
  if (std::getenv("VPIPE_QWEN4_EXP_F16") != nullptr) { precisions = {false}; }
  for (const bool bf16 : precisions) {
    MetalQwenModel::Config c = MetalQwenModel::config_from(*mcfg);
    c.use_bf16 = bf16;
    c.page_tokens = 256;
    c.max_pages = 64;
    c.max_seq = 8192;
    std::size_t cache_mb = 36864;
    if (const char* e = std::getenv("VPIPE_QWEN4_EXP_CACHE_MB")) {
      cache_mb = (std::size_t)std::atoll(e);
    }
    c.expert_cache_bytes = cache_mb << 20;
    auto m = MetalQwenModel::load(mp, mc, c);
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { return; }
    for (const std::string& g : goldens) {
      const auto ids = read_raw_<std::int32_t>(g + "/ids.i32");
      const auto pos = read_raw_<std::int32_t>(g + "/position_ids.i32");
      const auto img32 = read_raw_<float>(g + "/image_embeds_f32.f32");
      const auto ref = read_raw_<float>(g + "/logits_f32.f32");
      const auto ref16 = read_raw_<float>(g + "/logits_bf16.f32");
      const std::size_t T = ids.size();
      const std::size_t rows = V > 0 ? ref.size() / V : 0;
      ASSERT_TRUE(T > 0 && pos.size() == 3 * T && !img32.empty() &&
                  img32.size() % (std::size_t)H == 0 && rows > 0 &&
                  rows <= T && ref.size() == rows * V);
      if (T == 0 || pos.size() != 3 * T || img32.empty() || rows == 0 ||
          rows > T || ref.size() != rows * V) {
        continue;
      }
      const std::size_t n_img = img32.size() / (std::size_t)H;
      // The rows: text from the embedding table, the image's placeholders
      // the reference tower's rows in order.
      std::vector<std::int32_t> tids;
      for (const std::int32_t t : ids) {
        if (t != kImagePad) { tids.push_back(t); }
      }
      ASSERT_TRUE(T - tids.size() == n_img);
      if (T - tids.size() != n_img) { continue; }
      const std::vector<float> trows = m->embed_text(tids);
      ASSERT_TRUE(trows.size() == tids.size() * (std::size_t)H);
      if (trows.size() != tids.size() * (std::size_t)H) { continue; }
      std::vector<float> emb(T * (std::size_t)H);
      for (std::size_t t = 0, ti = 0, ii = 0; t < T; ++t) {
        const float* src = ids[t] == kImagePad ? img32.data() + (ii++) * H
                                               : trows.data() + (ti++) * H;
        std::memcpy(emb.data() + t * H, src, (std::size_t)H * 4);
      }
      // Prefilled in up to two calls, the kept rows' logits from the last.
      const auto cx = m->context_manager()->acquire_root();
      auto part = [&](std::size_t r0, std::size_t n, std::vector<float>* lg) {
        std::vector<float> e(emb.begin() + (long)(r0 * H),
                             emb.begin() + (long)((r0 + n) * H));
        std::vector<std::int32_t> id(ids.begin() + (long)r0,
                                     ids.begin() + (long)(r0 + n));
        std::vector<std::int32_t> p(3 * n);
        for (int a = 0; a < 3; ++a) {
          for (std::size_t t = 0; t < n; ++t) {
            p[(std::size_t)a * n + t] = pos[(std::size_t)a * T + r0 + t];
          }
        }
        return m->prefill_hc_spliced(cx, e, id, p, (int)n, lg);
      };
      const std::size_t n0 = T - rows;
      const auto t0 = std::chrono::steady_clock::now();
      if (n0 > 0) { ASSERT_TRUE(!part(0, n0, nullptr).empty()); }
      const std::string sel_path =
          std::string(std::getenv("TMPDIR") != nullptr ? std::getenv("TMPDIR")
                                                       : "/tmp") +
          "/qwen4_exp_image_qsa_sel.i32";
      std::remove(sel_path.c_str());
      setenv("VPIPE_QWEN_QSA_DUMP", sel_path.c_str(), 1);
      std::vector<float> got;
      ASSERT_TRUE(!part(n0, rows, &got).empty());
      unsetenv("VPIPE_QWEN_QSA_DUMP");
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
      m->forget_context(cx);
      m->context_manager()->release(cx);
      ASSERT_TRUE(got.size() == ref.size());
      if (got.size() != ref.size()) { continue; }
      int decisive = 0, agree = 0, agree16 = 0;
      const bool has16 = ref16.size() == ref.size();
      for (std::size_t t = 0; t < rows; ++t) {
        const float* r = ref.data() + t * V;
        const float* o = got.data() + t * V;
        std::size_t a = 0, ao = 0, a16 = 0, b = 0;
        for (std::size_t v = 1; v < V; ++v) {
          if (r[v] > r[a]) { a = v; }
          if (o[v] > o[ao]) { ao = v; }
          if (has16 && ref16[t * V + v] > ref16[t * V + a16]) { a16 = v; }
        }
        b = a == 0 ? 1 : 0;
        for (std::size_t v = 0; v < V; ++v) {
          if (v != a && r[v] > r[b]) { b = v; }
        }
        if (r[a] - r[b] > 0.5f) {
          ++decisive;
          agree += ao == a;
          agree16 += a16 == a;
        }
      }
      const double e = rel_l2_(ref.data(), got.data(), ref.size());
      const double bar =
          has16 ? rel_l2_(ref.data(), ref16.data(), ref.size()) : 0.0;
      // Past the QSA threshold, the last row's picks per full layer against
      // the reference's.
      int off = -1, off16 = -1;
      const auto sel = read_raw_<std::int32_t>(sel_path);
      const auto q32 = read_raw_<std::int32_t>(g + "/qsa_f32.i32");
      const auto q16 = read_raw_<std::int32_t>(g + "/qsa_bf16.i32");
      constexpr int kFull = 12;
      const bool sparse = T > 2051;
      if (sparse && !q32.empty() && q32.size() % kFull == 0) {
        const std::size_t W = q32.size() / kFull;
        auto picks = [&](const std::int32_t* p, std::size_t w) {
          std::vector<char> on(T, 0);
          for (std::size_t i = 0; i < w; ++i) {
            if (p[i] >= 0 && (std::size_t)p[i] < T) {
              on[(std::size_t)p[i]] = 1;
            }
          }
          return on;
        };
        auto differ = [&](const std::vector<char>& x,
                          const std::vector<char>& y) {
          int d = 0;
          for (std::size_t t = 0; t < T; ++t) { d += x[t] != y[t]; }
          return d / 2;
        };
        if (sel.size() >= kFull * (W + 1)) {
          off = 0;
          off16 = q16.size() == q32.size() ? 0 : -1;
          for (int f = 0; f < kFull; ++f) {
            const std::int32_t cnt = sel[(std::size_t)f * (W + 1) + W];
            const auto r = picks(q32.data() + (std::size_t)f * W, W);
            off += differ(picks(sel.data() + (std::size_t)f * (W + 1),
                                (std::size_t)std::max(cnt, 0)), r);
            if (off16 >= 0) {
              off16 += differ(picks(q16.data() + (std::size_t)f * W, W), r);
            }
          }
        }
      }
      std::printf("[qwen4_exp] image turn, %zu tokens (%zu image rows, %s) in "
                  "%.0f ms: %zu rows' logits %.4f from fp32 (reference bf16 "
                  "%.4f); decisive argmax %d/%d (reference bf16 %d)",
                  T, n_img, c.use_bf16 ? "bf16" : "f16", ms, rows, e, bar,
                  agree, decisive, agree16);
      if (sparse) {
        std::printf("; QSA picks %d tokens off fp32's (its bf16 %d)", off,
                    off16);
      }
      std::printf("\n");
      if (sparse) { EXPECT_TRUE(off >= 0); }
      if (has16) {
        if (bf16) {
          // bf16 the model's own, and a DRAW: the long turn's prefill laid
          // out as one piece reads 0.1309 / 35 decisive / 280 picks off,
          // as 2048 + 581 rows 0.1174 / 36 / 276 -- the same arithmetic,
          // each row as it is in either, its rounding carried another way
          // through 48 layers (in f16 the one piece is the closer: 0.0677
          // against 0.0831). Before, the experts' tier moved it as much
          // (grouped GEMV 0.1060 / 37, steel 0.1114 / 36).
          EXPECT_TRUE(e < 1.25 * bar);
          EXPECT_TRUE(agree + 5 >= agree16);
          EXPECT_TRUE(off16 < 0 || off <= 2 * off16);
        } else {
          // f16 the tight check: far closer to fp32 than the reference's
          // bf16 run. MUTATION, the indexer at text positions: the long
          // turn 0.1070 / 208 picks off (correct 0.0677 / 156; bf16 0.1470
          // against 0.1309).
          EXPECT_TRUE(e < 0.8 * bar);
          EXPECT_TRUE(agree + 4 >= agree16);
          EXPECT_TRUE(off16 < 0 || 4 * off <= 3 * off16);
        }
      }
    }
  }
}

// THE PLAN BOOKS WHAT THE LOAD HOLDS: the routed-expert cache is sized,
// and the LM's claim floored, at the checkpoint's bytes on disk less its
// experts and n-gram table plus what re-encoding the non-expert
// projections to w8 grows them by (held_growth_bytes). That figure against
// the device memory a load allocates beside its expert cache -- the vision
// tower, which the model does not hold but its stage does, set aside.
TEST(qwen4_exp, the_plan_books_what_the_load_holds)
{
  const char* mp = std::getenv("VPIPE_QWEN4_EXP_TEST_MODEL_PATH");
  if (mp == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  using vpipe::genai::MetalQwenModel;
  vpipe::genai::ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(mp);
  ASSERT_TRUE(mcfg.has_value());
  if (!mcfg.has_value()) { return; }
  MetalQwenModel::Config c = MetalQwenModel::config_from(*mcfg);
  c.use_bf16 = true;
  c.page_tokens = 256;
  c.max_pages = 16;
  c.max_seq = 4096;
  c.expert_cache_bytes = (std::size_t)2 << 30;
  auto w = vpipe::genai::MetalLlamaWeights::open_model(mp);
  ASSERT_TRUE(w.has_value());
  if (!w.has_value()) { return; }
  std::size_t disk = 0, vision = 0;
  for (const std::string& nm : w->tensor_names()) {
    const auto* ti = w->info(nm);
    if (ti == nullptr) { continue; }
    disk += (std::size_t)ti->nbytes;
    if (nm.rfind("vision", 0) == 0 ||
        nm.find(".visual.") != std::string::npos) {
      vision += (std::size_t)ti->nbytes;
    }
  }
  const std::size_t ex = MetalQwenModel::routed_expert_bytes(*w, c);
  const std::size_t un = MetalQwenModel::unheld_bytes(*w, c);
  const std::size_t gr = MetalQwenModel::held_growth_bytes(*w, c);
  const std::size_t plan = disk - ex - un - vision + gr;
  const std::size_t a0 = mc->memory_budget_settled().allocated;
  auto m = MetalQwenModel::load(mp, mc, c);
  ASSERT_TRUE(m != nullptr && m->expert_cache() != nullptr);
  if (m == nullptr || m->expert_cache() == nullptr) { return; }
  const std::size_t a1 = mc->memory_budget_settled().allocated;
  const std::size_t held = a1 - a0 - m->expert_cache()->bytes();
  std::printf("[qwen4_exp] %s: the plan books %.0f MB beside the experts "
              "(disk %.0f less experts %.0f, n-gram table %.0f, vision "
              "%.0f; re-encoded +%.0f); the load holds %.0f MB beside its "
              "%.0f MB cache\n", m->hc_layout().c_str(), plan / 1e6,
              disk / 1e6, ex / 1e6, un / 1e6, vision / 1e6, gr / 1e6,
              held / 1e6, m->expert_cache()->bytes() / 1e6);
  EXPECT_TRUE((double)held < 1.10 * (double)plan &&
              (double)held > 0.85 * (double)plan);
}

// THE PACKS Qwen3.8-Flash-Next is published in, each a small synthetic
// checkpoint in that pack's layout -- tools/make_qwen4_exp_synthetic.py
// --recipe R --out <dir>/R, then tools/dump_qwen4_exp_golden.py --model
// <dir>/R --out <dir>/R/golden --ids-file <dir>/R/ids.i32 --mtp -- under
// VPIPE_QWEN4_EXP_FLAVORS = <dir> (no published pack fits a box's disk
// beside another; the synthetic one is transformers' own modules at a size
// that still takes every path: hidden 2560 for the fused hyper-connection
// kernels, QSA past 67 tokens, the n-gram PLE, the MTP head). For each
// recipe present: the layout the load reports, a prefill's logits and the
// MTP head's against the fp32 reference (its bf16 run the bar), the decode
// path over the last rows, the draft head's greedy decode against the
// plain decode (token-exact, and its state the plain decode's to the bit),
// and the routed-expert cache against resident (bit-identical).
TEST(qwen4_exp_flavors, packs_match_reference)
{
  const char* fp = std::getenv("VPIPE_QWEN4_EXP_FLAVORS");
  if (fp == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  using vpipe::genai::MetalQwenModel;
  struct Want {
    const char* recipe;
    int group;
    const char* layout;    // what the load must report
  };
  const Want wants[] = {
      {"4bit-mtp", 32,
       "projections w8g32; router w8g32, experts w4g32, shared expert w8g32 "
       "(fused decode); MTP head: fc dense, attention dense, router dense, "
       "experts w4g32, shared expert dense"},
      {"oq5e", 64,
       "projections w8g64; router dense, experts w5g64, shared expert "
       "w8g64 (fused decode); MTP head: fc w8, attention w8"},
      {"mtplx", 32,
       "projections w8g32 (hyper-connections dense); router w8g32, experts "
       "w4g32"},
      {"g64", 64, "projections w8g64; router w8g64, experts w4g64"},
      {"w6", 64, "experts w6g64, shared expert w8g64 (fused decode)"},
      {"w8", 64, "experts w8g64, shared expert w8g64 (fused decode)"},
  };
  auto argmax = [](const float* v, std::size_t n) {
    std::size_t a = 0;
    for (std::size_t i = 1; i < n; ++i) {
      if (v[i] > v[a]) { a = i; }
    }
    return a;
  };
  int ran = 0;
  const char* only = std::getenv("VPIPE_QWEN4_EXP_FLAVOR");   // one recipe
  for (const Want& w : wants) {
    if (only != nullptr && std::string(only) != w.recipe) { continue; }
    const std::string mp = std::string(fp) + "/" + w.recipe;
    const std::string g = mp + "/golden";
    const auto ids = read_raw_<std::int32_t>(g + "/ids.i32");
    const auto ref = read_raw_<float>(g + "/logits_f32.f32");
    const auto ref16 = read_raw_<float>(g + "/logits_bf16.f32");
    const auto mref = read_raw_<float>(g + "/mtp_logits_f32.f32");
    const auto mref16 = read_raw_<float>(g + "/mtp_logits_bf16.f32");
    const std::size_t T = ids.size();
    if (T < 32 || ref.empty() || ref.size() % T != 0) { continue; }
    ++ran;
    const std::size_t V = ref.size() / T;
    vpipe::genai::ModelLoader loader(&sess);
    const auto mcfg = loader.load_config(mp);
    ASSERT_TRUE(mcfg.has_value());
    if (!mcfg.has_value()) { continue; }
    MetalQwenModel::Config c = MetalQwenModel::config_from(*mcfg);
    c.use_bf16 = std::getenv("VPIPE_QWEN4_EXP_F16") == nullptr;
    c.page_tokens = 256;
    c.max_pages = 16;
    c.max_seq = 4096;
    c.expert_cache_bytes = 0;
    EXPECT_TRUE(c.hc() && c.hc_group == w.group);
    // What the plan books for it, held whole: the checkpoint less its
    // n-gram table, plus the re-encode's growth.
    std::size_t plan = 0;
    if (auto wf = vpipe::genai::MetalLlamaWeights::open_model(mp)) {
      for (const std::string& nm : wf->tensor_names()) {
        if (const auto* ti = wf->info(nm)) { plan += (std::size_t)ti->nbytes; }
      }
      plan = plan - MetalQwenModel::unheld_bytes(*wf, c) +
             MetalQwenModel::held_growth_bytes(*wf, c);
    }
    const std::size_t a0 = mc->memory_budget_settled().allocated;
    auto m = MetalQwenModel::load(mp, mc, c);
    ASSERT_TRUE(m != nullptr && m->has_mtp());
    if (m == nullptr || !m->has_mtp()) { continue; }
    const std::size_t held = mc->memory_budget_settled().allocated - a0;
    const std::string lay = m->hc_layout();
    std::printf("[qwen4_exp_flavors] %s: %s; the plan books %.1f MB, the "
                "load holds %.1f\n", w.recipe, lay.c_str(), plan / 1e6,
                held / 1e6);
    EXPECT_TRUE(lay.find(w.layout) != std::string::npos);
    EXPECT_TRUE((double)held < 1.10 * (double)plan &&
                (double)held > 0.85 * (double)plan);

    // A prefill's logits, and the MTP head's teacher-forced pairs.
    const std::vector<std::int32_t> prompt(ids.begin(), ids.end());
    std::vector<float> mgot;
    const auto c0 = m->context_manager()->acquire_root();
    const std::vector<float> got = m->prefill_all_logits(c0, prompt, &mgot);
    m->context_manager()->release(c0);
    ASSERT_TRUE(got.size() == ref.size() && mgot.size() == mref.size());
    if (got.size() != ref.size() || mgot.size() != mref.size()) {
      continue;
    }
    int decisive = 0, agree = 0;
    for (std::size_t t = 0; t < T; ++t) {
      const float* r = ref.data() + t * V;
      const std::size_t a = argmax(r, V);
      float second = -1e30f;
      for (std::size_t v = 0; v < V; ++v) {
        if (v != a) { second = std::max(second, r[v]); }
      }
      if (r[a] - second > 0.5f) {
        ++decisive;
        agree += argmax(got.data() + t * V, V) == a;
      }
    }
    const double e = rel_l2_(ref.data(), got.data(), ref.size());
    const double bar = rel_l2_(ref.data(), ref16.data(), ref.size());
    const double me = rel_l2_(mref.data(), mgot.data(), mref.size());
    const double mbar = rel_l2_(mref.data(), mref16.data(), mref.size());
    std::printf("[qwen4_exp_flavors] %s: %zu-row prefill logits %.4f from "
                "fp32 (its bf16 %.4f), decisive argmax %d/%d; MTP head "
                "%.4f (%.4f)\n", w.recipe, T, e, bar, agree, decisive, me,
                mbar);
    EXPECT_TRUE(e < 1.5 * bar);
    EXPECT_TRUE(agree == decisive);
    EXPECT_TRUE(me < 1.5 * mbar);

    // The decode path: the last 16 rows a step each.
    {
      const std::size_t P = T - 16;
      const auto cd = m->context_manager()->acquire_root();
      std::vector<float> step = m->prefill(
          cd, std::vector<std::int32_t>(ids.begin(), ids.begin() + (long)P));
      std::vector<float> dg((T - P + 1) * V);
      ASSERT_TRUE(step.size() == V);
      if (step.size() != V) { continue; }
      std::memcpy(dg.data(), step.data(), V * 4);
      for (std::size_t t = P; t < T; ++t) {
        step = m->forward(cd, ids[t]);
        ASSERT_TRUE(step.size() == V);
        if (step.size() != V) { break; }
        std::memcpy(dg.data() + (t - P + 1) * V, step.data(), V * 4);
      }
      m->context_manager()->release(cd);
      const float* r0 = ref.data() + (P - 1) * V;
      const float* b0 = ref16.data() + (P - 1) * V;
      const double de = rel_l2_(r0, dg.data(), dg.size());
      const double dbar = rel_l2_(r0, b0, dg.size());
      // The row furthest from fp32 (a near-tie expert pick broken the
      // other way moves one row, not every one), ours and the reference's.
      double worst = 0.0, worst16 = 0.0;
      std::size_t wr = 0;
      for (std::size_t t = 0; t <= T - P; ++t) {
        const double et = rel_l2_(r0 + t * V, dg.data() + t * V, V);
        if (et > worst) { worst = et; wr = t; }
        worst16 = std::max(worst16, rel_l2_(r0 + t * V, b0 + t * V, V));
      }
      std::printf("[qwen4_exp_flavors] %s: decode over %zu rows %.4f "
                  "(%.4f); worst row %zu %.4f (its bf16's worst %.4f)\n",
                  w.recipe, T - P, de, dbar, wr, worst, worst16);
      // The reference's bf16 over 16 rows is one draw (w6's read 0.0088
      // where its whole prompt reads 0.0136, ours 0.012-0.014 on every
      // route and box -- f16 0.004): held to the larger of the two.
      EXPECT_TRUE(de < 1.5 * std::max(dbar, bar));
    }

    // The draft head's greedy decode against the plain one, and the state
    // it leaves (the next token's logits after it, to the bit).
    {
      constexpr int N = 32;
      const std::vector<std::int32_t> pp(ids.begin(), ids.begin() + 48);
      auto* cm = m->context_manager();
      const auto ca = cm->acquire_root();
      std::vector<float> l = m->prefill(ca, pp);
      std::vector<std::int32_t> plain{(std::int32_t)argmax(l.data(), V)};
      for (int i = 1; i <= N; ++i) {
        l = m->forward(ca, plain.back());
        plain.push_back((std::int32_t)argmax(l.data(), V));
      }
      const auto cb = cm->acquire_root();
      const std::int32_t first =
          (std::int32_t)argmax(m->prefill(cb, pp).data(), V);
      std::vector<std::int32_t> out;
      long kept = 0, rounds = 0;
      vpipe::genai::MtpDecodeCtl ctl;
      long drafted = 0;
      ctl.drafted = &drafted;
      // Three drafts a round, pinned: a random head's drafts are rarely
      // kept, and the measured count would mostly draft none -- the
      // replays of a partial accept are what this checks.
      ::setenv("VPIPE_QWEN_HC_MTP_DEPTH", "3", 1);
      const bool ok = m->mtp_decode(cb, first, N, out, 1, &kept, &rounds,
                                    ctl);
      ::unsetenv("VPIPE_QWEN_HC_MTP_DEPTH");
      ASSERT_TRUE(ok && (int)out.size() == N);
      int same = 0;
      for (int i = 0; i < (int)out.size() && i < N; ++i) {
        same += out[(std::size_t)i] == plain[(std::size_t)i];
      }
      // ca has taken plain[0 .. N - 1] (N forwards), as cb has out; the
      // next token's logits on both.
      const std::int32_t nxt = plain[(std::size_t)N];
      const int took_b = cm->seq_len_of(cb) - (int)pp.size();
      std::size_t diff = 0;
      if (took_b == N) {
        const std::vector<float> la = m->forward(ca, nxt);
        const std::vector<float> lb = m->forward(cb, nxt);
        for (std::size_t i = 0; i < V; ++i) { diff += la[i] != lb[i]; }
      }
      std::printf("[qwen4_exp_flavors] %s: draft head %d of %d tokens the "
                  "plain decode's, %ld of %ld drafts kept in %ld rounds; "
                  "the next logits after it: %zu of %zu differ\n", w.recipe,
                  same, N, kept, drafted, rounds, diff, V);
      EXPECT_TRUE(same == N);
      EXPECT_TRUE(took_b == N && diff == 0);
      EXPECT_TRUE(drafted > kept);          // a rejected draft: a replay
      cm->release(ca);
      cm->release(cb);
    }
    m.reset();

    // The routed experts from a cache a layer wide (every pick a read).
    {
      MetalQwenModel::Config cc = c;
      cc.expert_cache_bytes = (std::size_t)1 << 20;
      auto mx = MetalQwenModel::load(mp, mc, cc);
      ASSERT_TRUE(mx != nullptr);
      if (mx == nullptr) { continue; }
      std::vector<float> mx_mtp;
      const auto cx = mx->context_manager()->acquire_root();
      const std::vector<float> xg =
          mx->prefill_all_logits(cx, prompt, &mx_mtp);
      mx->context_manager()->release(cx);
      std::size_t diff = 0;
      for (std::size_t i = 0; i < xg.size() && i < got.size(); ++i) {
        diff += xg[i] != got[i];
      }
      for (std::size_t i = 0; i < mx_mtp.size() && i < mgot.size(); ++i) {
        diff += mx_mtp[i] != mgot[i];
      }
      std::printf("[qwen4_exp_flavors] %s: experts cached: %zu logits of "
                  "%zu differ from resident\n", w.recipe, diff,
                  got.size() + mgot.size());
      EXPECT_TRUE(xg.size() == got.size() && diff == 0);
    }
  }
  ASSERT_TRUE(ran > 0);
}
