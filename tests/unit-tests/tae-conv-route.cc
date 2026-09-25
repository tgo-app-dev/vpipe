// WHICH 3x3 ROUTE THE TAE's CONVS SHOULD TAKE ON MATRIX-CORE HARDWARE.
//
// A TAE preview is 98% its 3x3 gather conv (VPIPE_TAE_PROFILE, M5:
// conv3x3 63.9% + conv3x3+up 24.9% + head 9.5% on taeh3 at 960x576),
// so this one contraction is the whole question. Three ways to run it,
// at the shapes the published TAEs actually carry:
//
//   steel   conv3x3_gemm_s1_bn{32,64}_f16 -- the on-chip gather feeding
//           simdgroup MMA. What the decoder ships, and what carries the
//           fused ReLU / residual / upsample / concat epilogues.
//   mma2    conv2d_mma_3x3_s1_f16 -- the same on-chip gather feeding the
//           hardware matmul2d units.
//   hw      conv2d_hw_3x3_s1_f16 -- the MPP convolution2d op, which
//           gathers the window itself. It needs HWIO weights, has no
//           bias slot and cannot fuse an epilogue, so a win here has to
//           beat what those cost back; it also declines cout % 64.
//
// Env-gated (VPIPE_TAE_ROUTE_BENCH=1): it is a measurement, not a bar.
// Arms are interleaved per shape and the round order reversed on the
// second pass, because this box throttles across a long sweep and a
// sequential A-then-B reads the drift as a result.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::metal_compute;

namespace {

struct Shape { int cin, cout, H, W; const char* note; };

// One dispatch of `fn`, timed as the best of `reps` committed rounds.
double
time_one_(MetalCompute* mc, const ComputeFunction& fn, int reps,
          const std::function<void(ComputeEncoder&)>& encode)
{
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    CommandStream st = mc->make_command_stream();
    const auto t0 = std::chrono::steady_clock::now();
    { ComputeEncoder enc = st.begin_compute();
      enc.set_function(fn);
      encode(enc);
    }
    st.commit().wait();
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (r > 0) { best = std::min(best, ms); }   // r 0 warms
  }
  return best;
}

}  // namespace

TEST(tae_conv_route, steel_vs_matrix_cores)
{
  if (std::getenv("VPIPE_TAE_ROUTE_BENCH") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[tae-route] no matrix cores -- skip\n");
    return;
  }
  ComputeLibrary lg = mc->load_library("dense_gemm");
  ComputeLibrary lc = mc->load_library("conv2d_mma");
  ComputeFunction f_steel32 = lg.function("conv3x3_gemm_s1_bn32_f16");
  ComputeFunction f_steel64 = lg.function("conv3x3_gemm_s1_bn64_f16");
  ComputeFunction f_mma2    = lc.function("conv2d_mma_3x3_s1_f16");
  ComputeFunction f_hw      = lc.function("conv2d_hw_3x3_s1_f16");
  ASSERT_TRUE(f_steel32.valid() && f_steel64.valid() && f_mma2.valid());

  // taeh3 / taew2_1 at the two resolutions the decoder runs them: a
  // 960x576 video preview and a 1024^2 still. Each block's convs run at
  // the resolution its level reached, so the deep/narrow and
  // shallow/wide ends both appear.
  const Shape shapes[] = {
      {512, 256,  72, 120, "block-in"},
      {256, 256, 144, 240, "mid"},
      {256, 128, 144, 240, "narrow"},
      {128, 128, 288, 480, "wide"},
      {128,  64, 288, 480, "wide-narrow"},
      { 64,  64, 576, 960, "top"},
      { 64,  12, 576, 960, "head"},
      { 24, 256,  36,  60, "conv-in"},
  };

  std::printf("[tae-route] %-12s %-16s %9s %9s %9s %8s %8s\n", "shape",
              "geom", "steel ms", "mma2 ms", "hw ms", "st/mma2", "st/hw");
  for (const Shape& s : shapes) {
    const std::size_t nin  = (std::size_t)s.H * s.W * s.cin;
    const std::size_t nw   = (std::size_t)9 * s.cin * s.cout;
    const std::size_t nout = (std::size_t)s.H * s.W * s.cout;
    SharedBuffer ib = mc->make_shared_buffer(nin * 2);
    SharedBuffer wb = mc->make_shared_buffer(nw * 2);
    SharedBuffer bb = mc->make_shared_buffer((std::size_t)s.cout * 2);
    SharedBuffer ob = mc->make_shared_buffer(nout * 2);
    // The hardware op reads HWIO [9][cin][cout]; the other two read the
    // decoder's own [cout, 9*cin]. A twin, not a relayout.
    SharedBuffer hb = mc->make_shared_buffer(nw * 2);
    std::mt19937 rng(7u);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    auto* ip = static_cast<_Float16*>(ib.contents());
    for (std::size_t k = 0; k < nin; ++k) { ip[k] = (_Float16)d(rng); }
    auto* wp = static_cast<_Float16*>(wb.contents());
    for (std::size_t k = 0; k < nw; ++k) { wp[k] = (_Float16)(d(rng) * .2f); }
    std::memset(bb.contents(), 0, (std::size_t)s.cout * 2);
    auto* hp = static_cast<_Float16*>(hb.contents());
    for (int t = 0; t < 9; ++t) {
      for (int i = 0; i < s.cin; ++i) {
        for (int o = 0; o < s.cout; ++o) {
          hp[((std::size_t)t * s.cin + i) * s.cout + o] =
              wp[(std::size_t)o * 9 * s.cin + (std::size_t)t * s.cin + i];
        }
      }
    }

    const int bn = s.cout >= 64 ? 64 : 32;
    const int M = s.H * s.W;
    auto enc_steel = [&](ComputeEncoder& e) {
      e.set_buffer(0, ib); e.set_buffer(1, wb); e.set_buffer(2, bb);
      e.set_buffer(3, ob);
      e.set_constant(4, s.H);   e.set_constant(5, s.W);
      e.set_constant(6, s.cin); e.set_constant(7, s.cout);
      e.set_constant(8, s.H);   e.set_constant(9, s.W);
      e.set_constant(10, 1);
      e.dispatch({(unsigned)(((s.cout + bn - 1) / bn) * 32),
                  (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
    };
    auto enc_mma2 = [&](ComputeEncoder& e) {
      e.set_buffer(0, ib); e.set_buffer(1, wb); e.set_buffer(2, ob);
      e.set_constant(3, s.H);   e.set_constant(4, s.W);
      e.set_constant(5, s.cin); e.set_constant(6, s.cout);
      e.set_constant(7, s.H);   e.set_constant(8, s.W);
      e.dispatch({(unsigned)(((s.cout + 63) / 64) * 4 * 32),
                  (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
    };
    auto enc_hw = [&](ComputeEncoder& e) {
      e.set_buffer(0, ib); e.set_buffer(1, hb); e.set_buffer(2, ob);
      e.set_constant(3, s.W); e.set_constant(4, s.H);
      e.set_constant(5, s.cin); e.set_constant(6, s.cout);
      e.dispatch({(unsigned)((s.W / 8) * 128), (unsigned)(s.H / 8),
                  (unsigned)(s.cout / 64)}, {128, 1, 1});
    };
    // The op takes whole 8x8 dest tiles and 64-channel slices; the head
    // (cout 12) and a 60x36 grid are exactly what it declines.
    const bool hw_ok = f_hw.valid() && (s.W % 8) == 0 && (s.H % 8) == 0 &&
                       (s.cout % 64) == 0;
    const ComputeFunction& fs = bn == 64 ? f_steel64 : f_steel32;
    // Interleaved, then the round order reversed: the clamp does not
    // scale two arms equally, so a sequential sweep reads drift as a
    // result.
    const double a1 = time_one_(mc, fs,    5, enc_steel);
    const double b1 = time_one_(mc, f_mma2, 5, enc_mma2);
    const double b2 = time_one_(mc, f_mma2, 5, enc_mma2);
    const double a2 = time_one_(mc, fs,    5, enc_steel);
    // A TIME IS ONLY A RESULT IF THE ARM COMPUTED THE SAME THING. A
    // dispatch the driver declines is also very fast, and at 3x it is
    // exactly the reading that would be believed.
    double hw_rel = -1.0;
    if (hw_ok) {
      std::vector<float> ref(nout), got(nout);
      auto grab = [&](std::vector<float>* dst) {
        const auto* p = static_cast<const _Float16*>(ob.contents());
        for (std::size_t k = 0; k < nout; ++k) { (*dst)[k] = (float)p[k]; }
      };
      std::memset(ob.contents(), 0, nout * 2);
      time_one_(mc, fs, 1, enc_steel);
      grab(&ref);
      std::memset(ob.contents(), 0, nout * 2);
      time_one_(mc, f_hw, 1, enc_hw);
      grab(&got);
      double num = 0.0, den = 0.0;
      for (std::size_t k = 0; k < nout; ++k) {
        const double d = (double)got[k] - (double)ref[k];
        num += d * d; den += (double)ref[k] * (double)ref[k];
      }
      hw_rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
      EXPECT_TRUE(hw_rel < 3e-2);
    }
    double hw = 0.0;
    if (hw_ok) {
      const double c1 = time_one_(mc, f_hw, 5, enc_hw);
      const double c2 = time_one_(mc, f_hw, 5, enc_hw);
      hw = std::min(c1, c2);
    }
    const double st = std::min(a1, a2), mm = std::min(b1, b2);
    char geom[32];
    std::snprintf(geom, sizeof geom, "%dx%d %d->%d", s.W, s.H, s.cin,
                  s.cout);
    if (hw > 0.0) {
      std::printf("[tae-route] %-12s %-16s %9.3f %9.3f %9.3f %7.2fx "
                  "%7.2fx  (hw rel-L2 %.1e)\n", s.note, geom, st, mm, hw,
                  st / mm, st / hw, hw_rel);
    } else {
      std::printf("[tae-route] %-12s %-16s %9.3f %9.3f %9s %7.2fx %8s\n",
                  s.note, geom, st, mm, "declined", st / mm, "-");
    }
  }
}

// CAN THE CONCAT FORM TAKE THE HARDWARE OP AT ALL? A TAEHV MemBlock's
// first conv reads [x, past] -- two maps, Cin/2 each, concatenated on
// the channel axis and never materialized. The op gathers its own
// window from ONE activation, so the only way in is to split the conv
// in two: conv(x, W[.., :CS]) then conv(past, W[.., CS:]) ACCUMULATED
// onto it, which is the same trick the Wan VAE uses for its temporal
// taps. That costs a second dispatch and a second pass over the
// destination, so it is not obviously a win -- hence this, before any
// kernel gets written for it.
TEST(tae_conv_route, a_split_concat_conv_against_steel)
{
  if (std::getenv("VPIPE_TAE_ROUTE_BENCH") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  ComputeLibrary lg = mc->load_library("dense_gemm");
  ComputeLibrary lc = mc->load_library("conv2d_mma");
  ComputeFunction fs  = lg.function("conv3x3_gemm_s1_bn64_cat_relu_f16");
  ComputeFunction fh  = lc.function("conv2d_hw_3x3_s1_f16");
  ComputeFunction fha = lc.function("conv2d_hw_3x3_s1_acc_f16");
  ASSERT_TRUE(fs.valid() && fh.valid() && fha.valid());

  // The shapes a MemBlock's first conv actually takes: CS in, CS from
  // memory, at the level's resolution.
  struct S { int cs, cout, H, W; };
  const S shapes[] = {{64, 64, 1024, 1024}, {64, 64, 576, 960},
                      {128, 128, 288, 480}, {256, 256, 144, 240}};
  for (const S& s2 : shapes) {
    const int cin = 2 * s2.cs, M = s2.H * s2.W;
    const std::size_t nh = (std::size_t)M * s2.cs;
    const std::size_t nout = (std::size_t)M * s2.cout;
    SharedBuffer xa = mc->make_shared_buffer(nh * 2);
    SharedBuffer xb = mc->make_shared_buffer(nh * 2);
    const std::size_t nwf = (std::size_t)9 * cin * s2.cout;
    const std::size_t nwh = (std::size_t)9 * s2.cs * s2.cout;
    SharedBuffer wb = mc->make_shared_buffer(nwf * 2);
    SharedBuffer ha = mc->make_shared_buffer(nwh * 2);
    SharedBuffer hb2 = mc->make_shared_buffer(nwh * 2);
    SharedBuffer bb = mc->make_shared_buffer((std::size_t)s2.cout * 2);
    SharedBuffer ob = mc->make_shared_buffer(nout * 2);
    std::memset(bb.contents(), 0, (std::size_t)s2.cout * 2);
    auto steel = [&](ComputeEncoder& e) {
      e.set_buffer(0, xa); e.set_buffer(1, wb); e.set_buffer(2, bb);
      e.set_buffer(3, ob);
      e.set_constant(4, s2.H);  e.set_constant(5, s2.W);
      e.set_constant(6, cin);   e.set_constant(7, s2.cout);
      e.set_constant(8, s2.H);  e.set_constant(9, s2.W);
      e.set_constant(10, 1);
      e.set_buffer(11, ob); e.set_buffer(12, xb);
      e.dispatch({(unsigned)(((s2.cout + 63) / 64) * 32),
                  (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
    };
    // The split pair, timed as ONE encoder: that is how the decoder
    // would issue it, and a per-dispatch commit would measure the
    // commit.
    double best = 1e30, bs = 1e30;
    for (int r = 0; r < 5; ++r) {
      CommandStream st = mc->make_command_stream();
      const auto t0 = std::chrono::steady_clock::now();
      { ComputeEncoder e = st.begin_compute();
        for (int half = 0; half < 2; ++half) {
          e.set_function(half == 0 ? fh : fha);
          e.set_buffer(0, half == 0 ? xa : xb);
          e.set_buffer(1, half == 0 ? ha : hb2);
          e.set_buffer(2, ob);
          e.set_constant(3, s2.W);   e.set_constant(4, s2.H);
          e.set_constant(5, s2.cs);  e.set_constant(6, s2.cout);
          e.dispatch({(unsigned)((s2.W / 8) * 128), (unsigned)(s2.H / 8),
                      (unsigned)(s2.cout / 64)}, {128, 1, 1});
        }
      }
      st.commit().wait();
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
      if (r > 0) { best = std::min(best, ms); }
      const double t = time_one_(mc, fs, 2, steel);
      if (r > 0) { bs = std::min(bs, t); }
    }
    std::printf("[tae-cat] %4dx%-4d %3d+%3d->%3d  steel %8.3f ms  "
                "split-hw %8.3f ms  %.2fx\n", s2.W, s2.H, s2.cs, s2.cs,
                s2.cout, bs, best, bs / best);
  }
}

// IS THE FUSED 2x UPSAMPLE WORTH MATERIALIZING? The op reads its own
// window, so a conv whose input is nearest-upsampled on the fly has no
// way in -- the only route is to write the upsampled activation out
// first and convolve that, which costs a 4x buffer and a pass over it.
// [tae] a881677 folded that upsample INTO the gather conv precisely to
// avoid the pass, so this is the measurement that says whether the
// hardware op buys it back. The image TAEs are where it could: their up
// convs run at 512->1024, where the 3x kernel has the most to win
// against a fixed materialization cost.
TEST(tae_conv_route, a_materialized_upsample_against_the_fused_one)
{
  if (std::getenv("VPIPE_TAE_ROUTE_BENCH") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  ComputeLibrary lg = mc->load_library("dense_gemm");
  ComputeLibrary le = mc->load_library("llm_elementwise");
  ComputeLibrary lc = mc->load_library("conv2d_mma");
  ComputeFunction fs = lg.function("conv3x3_gemm_up2_bn64_relu_f16");
  ComputeFunction fu = le.function("upsample_nearest2x_hwc_f16");
  ComputeFunction fh = lc.function("conv2d_hw_3x3_s1_epi_relu_f16");
  ASSERT_TRUE(fs.valid() && fu.valid() && fh.valid());

  // input H x W at cin -> output 2H x 2W at cout, the shape an image
  // TAE's three up convs take.
  struct S { int cin, cout, H, W; };
  const S shapes[] = {{64, 64, 512, 512}, {64, 64, 256, 256},
                      {128, 64, 128, 128}, {64, 64, 288, 480}};
  for (const S& s2 : shapes) {
    const int OH = s2.H * 2, OW = s2.W * 2;
    const std::size_t nin = (std::size_t)s2.H * s2.W * s2.cin;
    const std::size_t nup = (std::size_t)OH * OW * s2.cin;
    const std::size_t nw = (std::size_t)9 * s2.cin * s2.cout;
    const std::size_t nout = (std::size_t)OH * OW * s2.cout;
    SharedBuffer ib = mc->make_shared_buffer(nin * 2);
    SharedBuffer ub = mc->make_shared_buffer(nup * 2);
    SharedBuffer wb = mc->make_shared_buffer(nw * 2);
    SharedBuffer hb = mc->make_shared_buffer(nw * 2);
    SharedBuffer bb = mc->make_shared_buffer((std::size_t)s2.cout * 2);
    SharedBuffer ob = mc->make_shared_buffer(nout * 2);
    std::memset(bb.contents(), 0, (std::size_t)s2.cout * 2);
    auto steel = [&](ComputeEncoder& e) {
      e.set_buffer(0, ib); e.set_buffer(1, wb); e.set_buffer(2, bb);
      e.set_buffer(3, ob);
      e.set_constant(4, s2.H);  e.set_constant(5, s2.W);
      e.set_constant(6, s2.cin); e.set_constant(7, s2.cout);
      e.set_constant(8, OH);    e.set_constant(9, OW);
      e.set_constant(10, 1);
      e.set_buffer(11, ob); e.set_buffer(12, ib);
      e.dispatch({(unsigned)(((s2.cout + 63) / 64) * 32),
                  (unsigned)((((OH * OW) + 63) / 64) * 2), 2}, {32, 2, 2});
    };
    double bs = 1e30, bh = 1e30;
    for (int r = 0; r < 5; ++r) {
      const double t = time_one_(mc, fs, 2, steel);
      if (r > 0) { bs = std::min(bs, t); }
      CommandStream st = mc->make_command_stream();
      const auto t0 = std::chrono::steady_clock::now();
      { ComputeEncoder e = st.begin_compute();
        e.set_function(fu);
        e.set_buffer(0, ib); e.set_buffer(1, ub);
        e.set_constant(2, s2.H); e.set_constant(3, s2.W);
        e.set_constant(4, s2.cin);
        e.dispatch({(unsigned)nup, 1, 1}, {256, 1, 1});
        e.set_function(fh);
        e.set_buffer(0, ub); e.set_buffer(1, hb); e.set_buffer(2, ob);
        e.set_constant(3, OW); e.set_constant(4, OH);
        e.set_constant(5, s2.cin); e.set_constant(6, s2.cout);
        e.set_buffer(7, bb); e.set_buffer(8, ob);
        e.dispatch({(unsigned)((OW / 8) * 128), (unsigned)(OH / 8),
                    (unsigned)(s2.cout / 64)}, {128, 1, 1});
      }
      st.commit().wait();
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
      if (r > 0) { bh = std::min(bh, ms); }
    }
    std::printf("[tae-up2] %4dx%-4d %3d->%3d  fused-gather %7.3f ms  "
                "materialize+hw %7.3f ms  %.2fx  (+%.1f MB)\n", OW, OH,
                s2.cin, s2.cout, bs, bh, bs / bh, (double)nup * 2 / 1e6);
  }
}

// THE FUSED HARDWARE CONVS MATCH THEIR STEEL TWINS, and are still worth
// taking after the epilogue. The plain hw op beat steel ~3x above, but
// that op has no bias slot and no epilogue -- so the comparison that
// decides anything is the FUSED pair, form for form. Not env-gated: an
// epilogue that silently stopped matching is a wrong picture, and these
// kernels are runtime-compiled, so a load failure only ever surfaces
// here.
TEST(tae_conv_route, fused_hardware_convs_match_steel)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[tae-fused] no matrix cores -- skip\n");
    return;
  }
  ComputeLibrary lg = mc->load_library("dense_gemm");
  ComputeLibrary lc = mc->load_library("conv2d_mma");
  struct Arm { const char* steel; const char* hw; int epi; };
  const Arm arms[] = {
      {"conv3x3_gemm_s1_bn64_f16", "conv2d_hw_3x3_s1_epi_f16", 0},
      {"conv3x3_gemm_s1_bn64_relu_f16",
       "conv2d_hw_3x3_s1_epi_relu_f16", 1},
      {"conv3x3_gemm_s1_bn64_addrelu_f16",
       "conv2d_hw_3x3_s1_epi_addrelu_f16", 2},
  };
  const int H = 144, W = 240, cin = 256, cout = 128, M = H * W;
  const std::size_t nin = (std::size_t)M * cin;
  const std::size_t nw = (std::size_t)9 * cin * cout;
  const std::size_t nout = (std::size_t)M * cout;
  SharedBuffer ib = mc->make_shared_buffer(nin * 2);
  SharedBuffer wb = mc->make_shared_buffer(nw * 2);
  SharedBuffer hb = mc->make_shared_buffer(nw * 2);
  SharedBuffer bb = mc->make_shared_buffer((std::size_t)cout * 2);
  SharedBuffer rb = mc->make_shared_buffer(nout * 2);
  SharedBuffer ob = mc->make_shared_buffer(nout * 2);
  std::mt19937 rng(11u);
  std::uniform_real_distribution<float> d(-0.5f, 0.5f);
  auto* ip = static_cast<_Float16*>(ib.contents());
  for (std::size_t k = 0; k < nin; ++k) { ip[k] = (_Float16)d(rng); }
  auto* wp = static_cast<_Float16*>(wb.contents());
  for (std::size_t k = 0; k < nw; ++k) { wp[k] = (_Float16)(d(rng) * .2f); }
  auto* bp = static_cast<_Float16*>(bb.contents());
  for (int k = 0; k < cout; ++k) { bp[k] = (_Float16)(d(rng) * .5f); }
  auto* rp = static_cast<_Float16*>(rb.contents());
  for (std::size_t k = 0; k < nout; ++k) { rp[k] = (_Float16)d(rng); }
  auto* hp = static_cast<_Float16*>(hb.contents());
  for (int t = 0; t < 9; ++t) {
    for (int i = 0; i < cin; ++i) {
      for (int o = 0; o < cout; ++o) {
        hp[((std::size_t)t * cin + i) * cout + o] =
            wp[(std::size_t)o * 9 * cin + (std::size_t)t * cin + i];
      }
    }
  }
  for (const Arm& a : arms) {
    ComputeFunction fs = lg.function(a.steel);
    ComputeFunction fh = lc.function(a.hw);
    // A runtime-compiled entry that did not build is INVALID, not
    // missing: without this the dispatch below would re-run whichever
    // kernel was bound last and quietly "match".
    ASSERT_TRUE(fs.valid());
    ASSERT_TRUE(fh.valid());
    if (!fs.valid() || !fh.valid()) { continue; }
    auto grab = [&](std::vector<float>* dst) {
      const auto* p = static_cast<const _Float16*>(ob.contents());
      dst->resize(nout);
      for (std::size_t k = 0; k < nout; ++k) { (*dst)[k] = (float)p[k]; }
    };
    std::vector<float> ref, got;
    std::memset(ob.contents(), 0, nout * 2);
    time_one_(mc, fs, 1, [&](ComputeEncoder& e) {
      e.set_buffer(0, ib); e.set_buffer(1, wb); e.set_buffer(2, bb);
      e.set_buffer(3, ob);
      e.set_constant(4, H);   e.set_constant(5, W);
      e.set_constant(6, cin); e.set_constant(7, cout);
      e.set_constant(8, H);   e.set_constant(9, W);
      e.set_constant(10, 1);
      if (a.epi != 0) { e.set_buffer(11, rb); e.set_buffer(12, ib); }
      e.dispatch({(unsigned)(((cout + 63) / 64) * 32),
                  (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
    });
    grab(&ref);
    std::memset(ob.contents(), 0, nout * 2);
    time_one_(mc, fh, 1, [&](ComputeEncoder& e) {
      e.set_buffer(0, ib); e.set_buffer(1, hb); e.set_buffer(2, ob);
      e.set_constant(3, W);   e.set_constant(4, H);
      e.set_constant(5, cin); e.set_constant(6, cout);
      e.set_buffer(7, bb);    e.set_buffer(8, rb);
      e.dispatch({(unsigned)((W / 8) * 128), (unsigned)(H / 8),
                  (unsigned)(cout / 64)}, {128, 1, 1});
    });
    grab(&got);
    double num = 0.0, den = 0.0;
    for (std::size_t k = 0; k < nout; ++k) {
      const double e = (double)got[k] - (double)ref[k];
      num += e * e; den += (double)ref[k] * (double)ref[k];
    }
    const double rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    // The plain op was ~3x steel, but this one pays a threadgroup
    // round-trip and (epi 2) a residual read. Timed here, interleaved
    // and with the round order reversed, because it is the fused pair
    // and not the plain one that the decoder would actually swap.
    auto steel = [&](ComputeEncoder& e) {
      e.set_buffer(0, ib); e.set_buffer(1, wb); e.set_buffer(2, bb);
      e.set_buffer(3, ob);
      e.set_constant(4, H);   e.set_constant(5, W);
      e.set_constant(6, cin); e.set_constant(7, cout);
      e.set_constant(8, H);   e.set_constant(9, W);
      e.set_constant(10, 1);
      if (a.epi != 0) { e.set_buffer(11, rb); e.set_buffer(12, ib); }
      e.dispatch({(unsigned)(((cout + 63) / 64) * 32),
                  (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
    };
    auto hw = [&](ComputeEncoder& e) {
      e.set_buffer(0, ib); e.set_buffer(1, hb); e.set_buffer(2, ob);
      e.set_constant(3, W);   e.set_constant(4, H);
      e.set_constant(5, cin); e.set_constant(6, cout);
      e.set_buffer(7, bb);    e.set_buffer(8, rb);
      e.dispatch({(unsigned)((W / 8) * 128), (unsigned)(H / 8),
                  (unsigned)(cout / 64)}, {128, 1, 1});
    };
    const double s1 = time_one_(mc, fs, 5, steel);
    const double h1 = time_one_(mc, fh, 5, hw);
    const double h2 = time_one_(mc, fh, 5, hw);
    const double s2 = time_one_(mc, fs, 5, steel);
    const double ts = std::min(s1, s2), th = std::min(h1, h2);
    std::printf("[tae-fused] epi %d  %-34s rel-L2 %.3e  steel %.3f ms  "
                "hw %.3f ms  %.2fx\n", a.epi, a.hw, rel, ts, th, ts / th);
    EXPECT_TRUE(rel < 3e-3);
  }
}
