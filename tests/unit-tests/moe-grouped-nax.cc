// The MoE grouped expert GEMMs on the M5 matrix cores
// (affine_qmm_grouped_nax.metal): the NAX twins of the steel grouped
// kernels, held to an f64 reference and to the steel twins over random
// codes at every routed-expert width (w4, w5, w8; group 64), f16 and bf16,
// at two experts' shapes (Qwen3.6-35B-A3B: hidden 2048 into 512;
// Qwen3.8-Flash-Next: 2560 into 640). Gate|up are interleaved by feature,
// as the loader lays a layer's slab; several experts with ragged pair
// counts, so pad slots and a pad tile are exercised.
//
// No checkpoint. Skips (and says so) on a GPU without matrix cores.
// VPIPE_MOE_NAX_BENCH=1 adds a timing of both twins at the 35B's real
// geometry (256 experts, top-8, 128..2048 tokens): interleaved arms, the
// median of 7 rounds, the expert GEMMs' TFLOP/s.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace vpipe;
using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::ComputeFunction;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

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

// One row of MLX affine codes at `bits` (K a multiple of 8): w4 two codes a
// byte, low nibble first; w8 a byte each; w5 runs of 8 codes in 40 bits,
// code j at bit 5j, little-endian.
void
pack_row_(const std::uint8_t* q, int K, int bits, std::uint8_t* out)
{
  if (bits == 8) {
    std::memcpy(out, q, (std::size_t)K);
  } else if (bits == 4) {
    for (int p = 0; p < K / 2; ++p) {
      out[p] = (std::uint8_t)((q[2 * p] & 15u) | ((q[2 * p + 1] & 15u) << 4));
    }
  } else {
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
}

struct Sorted {
  std::vector<std::int32_t> srow, sdst, t2e;
};

// Per expert, its pairs in pair order padded to whole 32-row tiles, then one
// pad tile (an expert id of -1) -- the sort's layout.
Sorted
sort_pairs_(const std::vector<std::int32_t>& eid, int E, int topk)
{
  constexpr int kTile = 32;
  Sorted s;
  std::vector<std::vector<int>> by(E);
  for (int p = 0; p < (int)eid.size(); ++p) { by[eid[p]].push_back(p); }
  for (int e = 0; e < E; ++e) {
    const auto& ps = by[e];
    const int tiles = ((int)ps.size() + kTile - 1) / kTile;
    for (int t = 0; t < tiles; ++t) { s.t2e.push_back(e); }
    for (int k = 0; k < tiles * kTile; ++k) {
      s.srow.push_back(k < (int)ps.size() ? ps[k] / topk : -1);
      s.sdst.push_back(k < (int)ps.size() ? ps[k] : -1);
    }
  }
  s.t2e.push_back(-1);
  for (int k = 0; k < kTile; ++k) {
    s.srow.push_back(-1);
    s.sdst.push_back(-1);
  }
  return s;
}

struct Twins {
  ComputeFunction gu_steel, dn_steel, gu_nax, dn_nax;
  bool valid() const
  {
    return gu_steel.valid() && dn_steel.valid() && gu_nax.valid() &&
           dn_nax.valid();
  }
};

Twins
load_twins_(MetalCompute* mc, int bits, bool bf16)
{
  const std::string sfx = bf16 ? "_bf16" : "";
  auto ls = mc->load_library("affine_qmm_steel" + sfx);
  auto ln = mc->load_library("affine_qmm_grouped_nax" + sfx);
  const std::string w = "_w" + std::to_string(bits) + "g64";
  return {ls.function("affine_qmm_grouped_swiglu" + w),
          ls.function("affine_qmm_grouped" + w),
          ln.function("affine_qmm_grouped_swiglu_nax" + w),
          ln.function("affine_qmm_grouped_nax" + w)};
}

// The two grouped GEMMs of one layer, as encode_moe_mlp_ dispatches them.
struct Layer {
  SharedBuffer wgu, sgu, bgu, wdn, sdn, bdn;
};

void
encode_pair_(ComputeEncoder& enc, const ComputeFunction& gu,
             const ComputeFunction& dn, bool nax, const Layer& w,
             const SharedBuffer& x, const SharedBuffer& gact,
             const SharedBuffer& part, const SharedBuffer& srow,
             const SharedBuffer& sdst, const SharedBuffer& t2e, int H, int I,
             int npad)
{
  const unsigned my = (unsigned)(((npad + 31) / 32) * 2);
  enc.set_function(gu);
  enc.set_buffer(0, w.wgu); enc.set_buffer(1, w.sgu);
  enc.set_buffer(2, w.bgu); enc.set_buffer(3, x);
  enc.set_buffer(4, gact);
  enc.set_constant(5, H); enc.set_constant(6, 2 * I);
  enc.set_constant(7, npad); enc.set_buffer(8, t2e);
  enc.set_buffer(9, srow);
  enc.dispatch({nax ? (unsigned)((2 * I / 64) * 32)
                    : (unsigned)(((2 * I + 31) / 32) * 32), my, 2},
               {32, 2, 2});
  enc.set_function(dn);
  enc.set_buffer(0, w.wdn); enc.set_buffer(1, w.sdn);
  enc.set_buffer(2, w.bdn); enc.set_buffer(3, gact);
  enc.set_buffer(4, part);
  enc.set_constant(5, I); enc.set_constant(6, H);
  enc.set_constant(7, npad); enc.set_buffer(8, t2e);
  enc.set_buffer(9, sdst);
  enc.dispatch({nax ? (unsigned)((H / 64) * 32)
                    : (unsigned)(((H + 31) / 32) * 32), my, 2},
               {32, 2, 2});
}

}  // namespace

TEST(moe_grouped_nax, twins_match_steel_and_f64)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[moe_nax] SKIP: no matrix cores on this GPU\n");
    return;
  }
  struct Shape { int H, I; const char* name; };
  const Shape shapes[] = {{2048, 512, "35B-A3B"}, {2560, 640, "Flash-Next"}};
  constexpr int E = 6, M = 48, TOPK = 4, G = 64, NP = M * TOPK;
  for (const Shape& sh : shapes) {
    const int H = sh.H, I = sh.I;
    for (const int bits : {4, 5, 8}) {
      std::uint32_t rng = 20261020u + (std::uint32_t)(bits * 977 + H);
      auto next = [&]() {
        rng = rng * 1664525u + 1013904223u;
        return rng;
      };
      auto unif = [&]() { return (next() >> 8) * (1.0 / 16777216.0); };
      std::vector<std::uint8_t> qgu((std::size_t)E * 2 * I * H),
          qdn((std::size_t)E * H * I);
      for (auto& c : qgu) { c = (std::uint8_t)(next() >> (32 - bits)); }
      for (auto& c : qdn) { c = (std::uint8_t)(next() >> (32 - bits)); }
      // Scales keep the weights' spread alike at every width; biases
      // centre the codes. Exact in f16 too at these magnitudes.
      const double s0 = 32.0 / (double)(1 << bits);
      auto mk_sb = [&](std::size_t n, std::vector<std::uint16_t>& s,
                       std::vector<std::uint16_t>& b) {
        s.resize(n);
        b.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
          const float sv = (float)(s0 * (2e-3 + 4e-3 * unif()));
          s[i] = to_bf16_(sv);
          b[i] = to_bf16_((float)(-(double)(1 << (bits - 1)) * sv *
                                  (0.8 + 0.4 * unif())));
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
        eid[p] = r < 20 ? r % 5 : 5;    // expert 5 rare: mostly padding
      }
      const std::size_t rb_gu = (std::size_t)H * bits / 8;
      const std::size_t rb_dn = (std::size_t)I * bits / 8;
      std::vector<std::uint8_t> wgu((std::size_t)E * 2 * I * rb_gu),
          wdn((std::size_t)E * H * rb_dn);
      for (std::size_t r = 0; r < (std::size_t)E * 2 * I; ++r) {
        pack_row_(&qgu[r * H], H, bits, &wgu[r * rb_gu]);
      }
      for (std::size_t r = 0; r < (std::size_t)E * H; ++r) {
        pack_row_(&qdn[r * I], I, bits, &wdn[r * rb_dn]);
      }
      auto deq = [](const std::vector<std::uint8_t>& q,
                    const std::vector<std::uint16_t>& s,
                    const std::vector<std::uint16_t>& b, std::size_t row,
                    int K, int k) {
        const std::size_t g = row * (K / G) + (std::size_t)(k / G);
        return from_bf16_(s[g]) * (double)q[row * K + k] + from_bf16_(b[g]);
      };
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
      const Sorted st = sort_pairs_(eid, E, TOPK);
      const int npad = (int)st.srow.size();
      auto upload = [&](const void* p, std::size_t n) {
        SharedBuffer b = mc->make_shared_buffer(n);
        std::memcpy(b.contents(), p, n);
        return b;
      };
      const SharedBuffer bsrow = upload(st.srow.data(), st.srow.size() * 4),
                         bsdst = upload(st.sdst.data(), st.sdst.size() * 4),
                         bt2e = upload(st.t2e.data(), st.t2e.size() * 4);
      for (const bool bf16 : {false, true}) {
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
          return bf16 ? from_bf16_(to_bf16_((float)d))
                      : from_f16_(to_f16_(d));
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
        Layer w{upload(wgu.data(), wgu.size()),
                upload(c_sgu.data(), c_sgu.size() * 2),
                upload(c_bgu.data(), c_bgu.size() * 2),
                upload(wdn.data(), wdn.size()),
                upload(c_sdn.data(), c_sdn.size() * 2),
                upload(c_bdn.data(), c_bdn.size() * 2)};
        const SharedBuffer bx = upload(c_x.data(), c_x.size() * 2);
        const Twins tw = load_twins_(mc, bits, bf16);
        ASSERT_TRUE(tw.valid());
        if (!tw.valid()) { return; }
        const std::size_t ga_bytes = (std::size_t)npad * I * 2,
                          pt_bytes = (std::size_t)NP * H * 2;
        SharedBuffer ga_s = mc->make_shared_buffer(ga_bytes),
                     pt_s = mc->make_shared_buffer(pt_bytes),
                     ga_n = mc->make_shared_buffer(ga_bytes),
                     pt_n = mc->make_shared_buffer(pt_bytes);
        // Poison the NAX outputs: a row the kernel never writes reads NaN.
        std::memset(ga_n.contents(), 0xff, ga_bytes);
        std::memset(pt_n.contents(), 0xff, pt_bytes);
        CommandStream cs = mc->make_command_stream();
        {
          ComputeEncoder enc = cs.begin_compute();
          encode_pair_(enc, tw.gu_steel, tw.dn_steel, false, w, bx, ga_s,
                       pt_s, bsrow, bsdst, bt2e, H, I, npad);
          encode_pair_(enc, tw.gu_nax, tw.dn_nax, true, w, bx, ga_n, pt_n,
                       bsrow, bsdst, bt2e, H, I, npad);
        }
        ASSERT_TRUE(cs.commit().wait_ok());
        // rel-L2 of a [pairs, W] output against a reference indexed the
        // same way; `sdst` maps a pair to its sorted slot (gact), or none.
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
        // The two twins against each other, element for element.
        auto rel_bufs = [&](const SharedBuffer& a, const SharedBuffer& b,
                            std::size_t n) {
          const auto* pa = static_cast<const std::uint16_t*>(a.contents());
          const auto* pb = static_cast<const std::uint16_t*>(b.contents());
          double num = 0.0, den = 0.0;
          for (std::size_t i = 0; i < n; ++i) {
            const double va = rd(pa[i]), vb = rd(pb[i]);
            num += (va - vb) * (va - vb);
            den += va * va;
          }
          return std::sqrt(num / den);
        };
        const double sa = rel(act_ref, ga_s, I, &st.sdst);
        const double sp = rel(part_ref, pt_s, H, nullptr);
        const double na = rel(act_ref, ga_n, I, &st.sdst);
        const double np_ = rel(part_ref, pt_n, H, nullptr);
        const double ns = rel_bufs(pt_s, pt_n, (std::size_t)NP * H);
        // Pad slots of the NAX gate|up are written (zeros), as steel's are.
        int pad_bad = 0;
        const auto* gn = static_cast<const std::uint16_t*>(ga_n.contents());
        for (std::size_t k = 0; k < st.srow.size(); ++k) {
          if (st.srow[k] >= 0) { continue; }
          const bool tile_live = st.t2e[k / 32] >= 0;
          for (int c = 0; tile_live && c < I; ++c) {
            pad_bad += rd(gn[k * (std::size_t)I + c]) != 0.0;
          }
        }
        std::printf("[moe_nax] %s w%d %s vs f64: steel act %.2e part %.2e, "
                    "NAX act %.2e part %.2e; NAX vs steel %.2e; pad slots "
                    "not zero %d\n", sh.name, bits, bf16 ? "bf16" : "f16",
                    sa, sp, na, np_, ns, pad_bad);
        // Storage rounding of act / part is ~2e-3 at bf16 and ~3e-4 at
        // f16; a wrong code, slab, row or lane is O(1).
        const double bar = bf16 ? 6e-3 : 1.5e-3;
        EXPECT_TRUE(sa < bar && sp < bar);
        EXPECT_TRUE(na < bar && np_ < bar);
        EXPECT_TRUE(ns < bar);
        EXPECT_TRUE(pad_bad == 0);
      }
    }
  }
}

// VPIPE_MOE_NAX_BENCH=1: both twins at Qwen3.6-35B-A3B's geometry (256
// experts top-8, hidden 2048, inner 512, w4) over random routes.
TEST(moe_grouped_nax, bench_against_steel)
{
  const char* on = std::getenv("VPIPE_MOE_NAX_BENCH");
  if (on == nullptr || std::atoi(on) == 0) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) {
    std::printf("[moe_nax] bench SKIP: no matrix cores\n");
    return;
  }
  constexpr int E = 256, TOPK = 8, H = 2048, I = 512, BITS = 4;
  std::uint32_t rng = 7u;
  auto next = [&]() {
    rng = rng * 1664525u + 1013904223u;
    return rng;
  };
  auto rnd_bytes = [&](std::size_t n) {
    SharedBuffer b = mc->make_shared_buffer(n);
    auto* p = static_cast<std::uint8_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) { p[i] = (std::uint8_t)(next() >> 24); }
    return b;
  };
  auto scal = [&](std::size_t n, float lo, float span) {
    SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* p = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      p[i] = to_f16_(lo + span * (double)(next() >> 8) / 16777216.0);
    }
    return b;
  };
  Layer w{rnd_bytes((std::size_t)E * 2 * I * H * BITS / 8),
          scal((std::size_t)E * 2 * I * (H / 64), 2e-3f, 4e-3f),
          scal((std::size_t)E * 2 * I * (H / 64), -0.04f, 0.02f),
          rnd_bytes((std::size_t)E * H * I * BITS / 8),
          scal((std::size_t)E * H * (I / 64), 2e-3f, 4e-3f),
          scal((std::size_t)E * H * (I / 64), -0.04f, 0.02f)};
  const Twins tw = load_twins_(mc, BITS, false);
  ASSERT_TRUE(tw.valid());
  if (!tw.valid()) { return; }
  for (const int M : {128, 256, 512, 2048}) {
    const int NP = M * TOPK;
    std::vector<std::int32_t> eid(NP);
    for (int t = 0; t < M; ++t) {
      for (int k = 0; k < TOPK; ++k) {
        int e;
        bool dup;
        do {
          e = (int)(next() % (unsigned)E);
          dup = false;
          for (int j = 0; j < k; ++j) { dup |= eid[t * TOPK + j] == e; }
        } while (dup);
        eid[t * TOPK + k] = e;
      }
    }
    const Sorted st = sort_pairs_(eid, E, TOPK);
    const int npad = (int)st.srow.size();
    auto up = [&](const std::vector<std::int32_t>& v) {
      SharedBuffer b = mc->make_shared_buffer(v.size() * 4);
      std::memcpy(b.contents(), v.data(), v.size() * 4);
      return b;
    };
    const SharedBuffer srow = up(st.srow), sdst = up(st.sdst),
                       t2e = up(st.t2e);
    const SharedBuffer x = scal((std::size_t)M * H, -1.0f, 2.0f);
    SharedBuffer gact = mc->make_shared_buffer((std::size_t)npad * I * 2),
                 part = mc->make_shared_buffer((std::size_t)NP * H * 2);
    CommandStream cs = mc->make_command_stream();
    constexpr int kReps = 8;
    auto time_arm = [&](bool nax) {
      {
        ComputeEncoder enc = cs.begin_compute();
        for (int r = 0; r < kReps; ++r) {
          encode_pair_(enc, nax ? tw.gu_nax : tw.gu_steel,
                       nax ? tw.dn_nax : tw.dn_steel, nax, w, x, gact, part,
                       srow, sdst, t2e, H, I, npad);
        }
      }
      const auto t0 = std::chrono::steady_clock::now();
      const bool ok = cs.commit().wait_ok();
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
      return ok ? ms / kReps : -1.0;
    };
    time_arm(false);
    time_arm(true);                       // warm both
    std::vector<double> ts, tn;
    for (int round = 0; round < 7; ++round) {
      if (round % 2 == 0) {
        ts.push_back(time_arm(false));
        tn.push_back(time_arm(true));
      } else {
        tn.push_back(time_arm(true));
        ts.push_back(time_arm(false));
      }
    }
    std::sort(ts.begin(), ts.end());
    std::sort(tn.begin(), tn.end());
    const double flop = 2.0 * NP * (2.0 * I * H + (double)H * I);
    std::printf("[moe_nax] bench %4d tokens (%5d pairs, %4d tiles): steel "
                "%.3f ms (%.2f TFLOP/s), NAX %.3f ms (%.2f TFLOP/s), %.2fx\n",
                M, NP, (int)st.t2e.size(), ts[3], flop / ts[3] * 1e-9,
                tn[3], flop / tn[3] * 1e-9, ts[3] / tn[3]);
  }
}
