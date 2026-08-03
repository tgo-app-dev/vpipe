// quant-gemv.cc -- qmv (decode-GEMV) correctness: the affine-quantised matvec
// kernels (w4 group-32, the batched/tail variants, the fused GEGLU form)
// checked against CPU references, plus the threadgroup-vs-batch equivalence
// probe.

#include "tests/unit-tests/metal-lm/metal-lm-test-common.h"

// The tgmem-staged tall-tile batched GEMV (affine_qmv_batch8_tg*_w4g64,
// MAXM=8: activations staged in threadgroup memory instead of per-thread
// registers) must be BYTE-IDENTICAL to the proven register kernel
// (affine_qmv_batch_w4g64, the token-exact-verified batched-decode path)
// on the same inputs -- including a partial K tail (K=768 -> 256-tail
// block) and a padded row tile (m=7 < MAXM=8). Always-on (no model).
TEST(metal_lm_smoke, qmv_batch_tg_matches_batch) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("affine_qmv");
  struct V { const char* fn; int bits; int nsg; int maxm; };
  const V vars[] = {
      {"affine_qmv_batch8_tg_w4g64", 4, 2, 8},
      {"affine_qmv_batch8_tg4_w4g64", 4, 4, 8},
      {"affine_qmv_batch8_tg4_w8g64", 8, 4, 8},
      {"affine_qmv_batch8_xd_w4g64", 4, 2, 8},
      {"affine_qmv_batch8_xd_w8g64", 8, 2, 8},
      {"affine_qmv_batch8_xp1_w4g64", 4, 2, 8},
      {"affine_qmv_batch8_xp2_w4g64", 4, 2, 8},
      {"affine_qmv_batch8_xh_w4g64", 4, 2, 8},
      {"affine_qmv_batch4_xp_w4g64", 4, 2, 4},
      {"affine_qmv_batch4_xp_w8g64", 8, 2, 4},
      {"affine_qmv_batch8_xp2_w8g64", 8, 2, 8},
  };
  const int m = 7, N = 64, K = 768;   // K%512 != 0: exercises the tail path
  const int groups = K / 64;
  std::mt19937 rng(11);
  std::uniform_int_distribution<std::uint32_t> du(0, 0xffffffffu);
  std::uniform_real_distribution<float> df(-1.0f, 1.0f);
  for (const V& v : vars) {
    auto ref_fn = lib.function(
        v.bits == 4 ? "affine_qmv_batch_w4g64" : "affine_qmv_batch_w8g64");
    auto tg_fn = lib.function(v.fn);
    ASSERT_TRUE(ref_fn.valid());
    ASSERT_TRUE(tg_fn.valid());
    const std::size_t wwords = (std::size_t)N * K / (32 / v.bits);
    auto wb = mc->make_shared_buffer(wwords * 4);
    auto sb = mc->make_shared_buffer((std::size_t)N * groups * 2);
    auto bb = mc->make_shared_buffer((std::size_t)N * groups * 2);
    auto xb = mc->make_shared_buffer((std::size_t)m * K * 2);
    auto y0 = mc->make_shared_buffer((std::size_t)m * N * 2);
    auto y1 = mc->make_shared_buffer((std::size_t)m * N * 2);
    auto* wp = static_cast<std::uint32_t*>(wb.contents());
    for (std::size_t i = 0; i < wwords; ++i) { wp[i] = du(rng); }
    auto fill_h = [&](metal_compute::SharedBuffer& b, std::size_t n) {
      auto* p = static_cast<__fp16*>(b.contents());
      for (std::size_t i = 0; i < n; ++i) { p[i] = (__fp16)(df(rng) * 0.5f); }
    };
    fill_h(sb, (std::size_t)N * groups);
    fill_h(bb, (std::size_t)N * groups);
    fill_h(xb, (std::size_t)m * K);
    auto run = [&](metal_compute::ComputeFunction& fn,
                   metal_compute::SharedBuffer& y, unsigned maxm,
                   unsigned nsg) {
      metal_compute::CommandStream st = mc->make_command_stream();
      {
        metal_compute::ComputeEncoder e = st.begin_compute();
        e.set_function(fn);
        e.set_buffer(0, wb); e.set_buffer(1, sb); e.set_buffer(2, bb);
        e.set_buffer(3, xb); e.set_buffer(4, y);
        e.set_constant(5, K); e.set_constant(6, N); e.set_constant(7, m);
        e.dispatch({32u, (unsigned)(N / 4), (unsigned)((m + maxm - 1) / maxm)},
                   {32u, nsg, 1u});
      }
      st.commit().wait();
    };
    run(ref_fn, y0, 2, 2);
    run(tg_fn, y1, (unsigned)v.maxm, (unsigned)v.nsg);
    const bool same = std::memcmp(y0.contents(), y1.contents(),
                                  (std::size_t)m * N * 2) == 0;
    if (!same) {
      const auto* a = static_cast<const __fp16*>(y0.contents());
      const auto* b2 = static_cast<const __fp16*>(y1.contents());
      int bad = 0;
      for (int i = 0; i < m * N && bad < 5; ++i) {
        if ((float)a[i] != (float)b2[i]) {
          std::printf("[qmv-batch-tg] %s mismatch @%d: %f vs %f\n",
                      v.fn, i, (float)a[i], (float)b2[i]);
          ++bad;
        }
      }
    }
    std::printf("[qmv-batch-tg] %-28s vs batch(MAXM=2): %s\n", v.fn,
                same ? "byte-identical" : "MISMATCH");
    EXPECT_TRUE(same);
  }

  // Fused-SwiGLU xp twins vs the register batch-swiglu (MAXM=2) reference:
  // interleaved gate/up weights, halved [m, N/2] output.
  {
    auto ref_fn = lib.function("affine_qmv_batch_swiglu_w4g64");
    ASSERT_TRUE(ref_fn.valid());
    struct SV { const char* fn; int maxm; };
    const SV svars[] = {
        {"affine_qmv_batch4_xp_swiglu_w4g64", 4},
        {"affine_qmv_batch8_xp2_swiglu_w4g64", 8},
    };
    const std::size_t wwords = (std::size_t)N * K / 8;
    auto wb = mc->make_shared_buffer(wwords * 4);
    auto sb = mc->make_shared_buffer((std::size_t)N * groups * 2);
    auto bb = mc->make_shared_buffer((std::size_t)N * groups * 2);
    auto xb = mc->make_shared_buffer((std::size_t)m * K * 2);
    auto y0 = mc->make_shared_buffer((std::size_t)m * (N / 2) * 2);
    auto y1 = mc->make_shared_buffer((std::size_t)m * (N / 2) * 2);
    auto* wp = static_cast<std::uint32_t*>(wb.contents());
    for (std::size_t i = 0; i < wwords; ++i) { wp[i] = du(rng); }
    auto fill_h = [&](metal_compute::SharedBuffer& b, std::size_t n) {
      auto* p = static_cast<__fp16*>(b.contents());
      for (std::size_t i = 0; i < n; ++i) { p[i] = (__fp16)(df(rng) * 0.5f); }
    };
    fill_h(sb, (std::size_t)N * groups);
    fill_h(bb, (std::size_t)N * groups);
    fill_h(xb, (std::size_t)m * K);
    auto run = [&](metal_compute::ComputeFunction& fn,
                   metal_compute::SharedBuffer& y, unsigned maxm) {
      metal_compute::CommandStream st = mc->make_command_stream();
      {
        metal_compute::ComputeEncoder e = st.begin_compute();
        e.set_function(fn);
        e.set_buffer(0, wb); e.set_buffer(1, sb); e.set_buffer(2, bb);
        e.set_buffer(3, xb); e.set_buffer(4, y);
        e.set_constant(5, K); e.set_constant(6, N); e.set_constant(7, m);
        e.dispatch({32u, (unsigned)(N / 4), (unsigned)((m + maxm - 1) / maxm)},
                   {32u, 2u, 1u});
      }
      st.commit().wait();
    };
    run(ref_fn, y0, 2);
    for (const SV& v : svars) {
      auto fn = lib.function(v.fn);
      ASSERT_TRUE(fn.valid());
      run(fn, y1, (unsigned)v.maxm);
      const bool same = std::memcmp(y0.contents(), y1.contents(),
                                    (std::size_t)m * (N / 2) * 2) == 0;
      std::printf("[qmv-batch-tg] %-32s vs batch_swiglu: %s\n", v.fn,
                  same ? "byte-identical" : "MISMATCH");
      EXPECT_TRUE(same);
    }
  }

  // xh16 (half-precision products/quad-sums, f32 block accumulator) is NOT
  // bit-identical by design; check it stays within a tight rel tolerance of
  // the reference (the token-exact bar is enforced end-to-end by
  // qwen_batched_decode_token_exact).
  {
    auto ref_fn = lib.function("affine_qmv_batch_w4g64");
    auto h_fn = lib.function("affine_qmv_batch8_xh16_w4g64");
    ASSERT_TRUE(h_fn.valid());
    const std::size_t wwords = (std::size_t)N * K / 8;
    auto wb = mc->make_shared_buffer(wwords * 4);
    auto sb = mc->make_shared_buffer((std::size_t)N * groups * 2);
    auto bb = mc->make_shared_buffer((std::size_t)N * groups * 2);
    auto xb = mc->make_shared_buffer((std::size_t)m * K * 2);
    auto y0 = mc->make_shared_buffer((std::size_t)m * N * 2);
    auto y1 = mc->make_shared_buffer((std::size_t)m * N * 2);
    auto* wp = static_cast<std::uint32_t*>(wb.contents());
    for (std::size_t i = 0; i < wwords; ++i) { wp[i] = du(rng); }
    auto fill_h = [&](metal_compute::SharedBuffer& b, std::size_t n) {
      auto* p = static_cast<__fp16*>(b.contents());
      for (std::size_t i = 0; i < n; ++i) { p[i] = (__fp16)(df(rng) * 0.5f); }
    };
    fill_h(sb, (std::size_t)N * groups);
    fill_h(bb, (std::size_t)N * groups);
    fill_h(xb, (std::size_t)m * K);
    auto run = [&](metal_compute::ComputeFunction& fn,
                   metal_compute::SharedBuffer& y, unsigned maxm) {
      metal_compute::CommandStream st = mc->make_command_stream();
      {
        metal_compute::ComputeEncoder e = st.begin_compute();
        e.set_function(fn);
        e.set_buffer(0, wb); e.set_buffer(1, sb); e.set_buffer(2, bb);
        e.set_buffer(3, xb); e.set_buffer(4, y);
        e.set_constant(5, K); e.set_constant(6, N); e.set_constant(7, m);
        e.dispatch({32u, (unsigned)(N / 4), (unsigned)((m + maxm - 1) / maxm)},
                   {32u, 2u, 1u});
      }
      st.commit().wait();
    };
    run(ref_fn, y0, 2);
    run(h_fn, y1, 8);
    const auto* a = static_cast<const __fp16*>(y0.contents());
    const auto* b2 = static_cast<const __fp16*>(y1.contents());
    double num = 0, den = 0;
    for (int i = 0; i < m * N; ++i) {
      const double d = (double)a[i] - (double)b2[i];
      num += d * d; den += (double)a[i] * (double)a[i];
    }
    const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
    std::printf("[qmv-batch-tg] affine_qmv_batch8_xh16_w4g64 rel-L2 %.3e\n",
                rel);
    EXPECT_TRUE(rel < 2e-3);
  }
}

// Numerically verify the decode-only quantized GEMV at group-size 32
// (affine_qmv_w4g32 -- the GGUF q4_0 path's q/k/v/o + down_proj decode
// kernel, never exercised by prefill [steel qmm] or e4b [g64]) against a
// hand-built CPU dequant+matmul reference. No model/MLX needed.
TEST(metal_lm_smoke, qmv_w4g32_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("affine_qmv");
  auto fn = lib.function("affine_qmv_w4g32");
  ASSERT_TRUE(fn.valid());

  // Tail guard: K = 3840 (Gemma-12B hidden) is NOT a multiple of the 4-bit
  // block_size (512) -> the GEMV's final block is partial. We allocate x out
  // to the block boundary (Kpad = 4096) filled with non-zero values, the
  // weights with one extra padding row, and tell the kernel in_vec_size = K.
  // The reference sums only k < K. WITHOUT the tail mask the GPU's tail lanes
  // (k in [3840,4096)) fold non-zero x * out-of-range weights into the dot ->
  // gross error; WITH the mask they contribute 0 and it matches the CPU ref.
  // Uses small biases (no affine cancellation) so this isolates the tail.
  const int N = 16, K = 3840, G = 32, groups = K / G;
  const int Kpad = ((K + 511) / 512) * 512;        // 4096
  std::vector<std::uint32_t> wq((std::size_t)(N + 1) * (K / 8), 0);
  std::vector<std::uint16_t> q((std::size_t)N * K);
  std::vector<_Float16> scales((std::size_t)N * groups);
  std::vector<_Float16> biases((std::size_t)N * groups);
  std::vector<_Float16> x((std::size_t)Kpad);
  auto frand = [](unsigned i) {
    return (float)((i * 2654435761u) >> 9 & 0x3ff) / 1023.0f - 0.5f;
  };
  for (int o = 0; o < N; ++o) {
    for (int g = 0; g < groups; ++g) {
      scales[(std::size_t)o * groups + g] =
          (_Float16)(0.03f + 0.02f * frand((unsigned)(o * 991 + g)));
      biases[(std::size_t)o * groups + g] =
          (_Float16)(0.1f * frand((unsigned)(o * 131 + g * 7 + 5)));
    }
  }
  // Fill the whole weight buffer (incl. the extra padding row) so tail lanes
  // read deterministic non-zero nibbles when the mask is absent.
  for (int o = 0; o <= N; ++o) {
    for (int k = 0; k < K; ++k) {
      const unsigned qq = ((unsigned)(o * K + k) * 2654435761u >> 13) & 0xf;
      if (o < N) { q[(std::size_t)o * K + k] = (std::uint16_t)qq; }
      wq[(std::size_t)o * (K / 8) + k / 8] |=
          (std::uint32_t)qq << (4 * (k % 8));
    }
  }
  for (int k = 0; k < Kpad; ++k) {
    x[k] = (_Float16)(0.5f + frand((unsigned)(k * 7 + 3)));   // all non-zero
  }

  std::vector<float> ref((std::size_t)N);
  for (int o = 0; o < N; ++o) {
    double acc = 0.0;
    for (int k = 0; k < K; ++k) {
      const float s = (float)scales[(std::size_t)o * groups + k / G];
      const float b = (float)biases[(std::size_t)o * groups + k / G];
      acc += (double)(float)x[k] *
             (s * (float)q[(std::size_t)o * K + k] + b);
    }
    ref[o] = (float)acc;
  }

  auto wbuf = mc->make_shared_buffer(wq.size() * 4);
  auto sbuf = mc->make_shared_buffer(scales.size() * 2);
  auto bbuf = mc->make_shared_buffer(biases.size() * 2);
  auto xbuf = mc->make_shared_buffer(x.size() * 2);
  auto ybuf = mc->make_shared_buffer((std::size_t)N * 2);
  std::memcpy(wbuf.contents(), wq.data(), wq.size() * 4);
  std::memcpy(sbuf.contents(), scales.data(), scales.size() * 2);
  std::memcpy(bbuf.contents(), biases.data(), biases.size() * 2);
  std::memcpy(xbuf.contents(), x.data(), x.size() * 2);
  auto stream = mc->make_command_stream();
  {
    auto enc = stream.begin_compute();
    enc.set_function(fn);
    enc.set_buffer(0, wbuf);
    enc.set_buffer(1, sbuf);
    enc.set_buffer(2, bbuf);
    enc.set_buffer(3, xbuf);
    enc.set_buffer(4, ybuf);
    enc.set_constant(5, K);
    enc.set_constant(6, N);
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
  }
  stream.commit().wait();
  const auto* yp = static_cast<const _Float16*>(ybuf.contents());
  double maxrel = 0.0;
  for (int o = 0; o < N; ++o) {
    const double d = std::fabs((double)(float)yp[o] - (double)ref[o]);
    maxrel = std::fmax(maxrel, d / (std::fabs((double)ref[o]) + 1e-2));
    std::printf("[qmv_w4g32] o=%2d gpu=%.5f ref=%.5f\n",
                o, (float)yp[o], ref[o]);
  }
  std::printf("[qmv_w4g32] max rel err = %.4g\n", maxrel);
  EXPECT_TRUE(maxrel < 0.03);
}

// Batched (MAXM=2) decode GEMV at group-size 32 with K = 3840 (NOT a multiple
// of block_size 512 -- Gemma-12B hidden): exercises the partial-tail path in
// qmv_batch_impl. Two rows, verified against a CPU dequant ref per (row,out).
TEST(metal_lm_smoke, qmv_batch_w4g32_tail_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("affine_qmv");
  auto fn = lib.function("affine_qmv_batch_w4g32");
  ASSERT_TRUE(fn.valid());

  // Tail guard: K = 3840 (not a multiple of block_size 512). Each row's x is
  // padded to the block boundary (Kpad) with non-zero values and the weights
  // carry an extra padding row, so the partial last block reads out-of-range
  // data unless masked. Small biases -> isolates the tail (no cancellation).
  const int N = 16, M = 2, K = 3840, G = 32, groups = K / G;
  const int Kpad = ((K + 511) / 512) * 512;        // 4096
  std::vector<std::uint32_t> wq((std::size_t)(N + 1) * (K / 8), 0);
  std::vector<std::uint16_t> q((std::size_t)N * K);
  std::vector<_Float16> scales((std::size_t)N * groups);
  std::vector<_Float16> biases((std::size_t)N * groups);
  // x rows are at the kernel's stride (in_vec_size = K), with trailing padding
  // so the last row's partial-block tail reads in-bounds (non-zero) data.
  std::vector<_Float16> x((std::size_t)M * K + (Kpad - K));
  auto frand = [](unsigned i) {
    return (float)((i * 2654435761u) >> 9 & 0x3ff) / 1023.0f - 0.5f;
  };
  for (int o = 0; o < N; ++o) {
    for (int g = 0; g < groups; ++g) {
      scales[(std::size_t)o * groups + g] =
          (_Float16)(0.03f + 0.02f * frand((unsigned)(o * 991 + g)));
      biases[(std::size_t)o * groups + g] =
          (_Float16)(0.1f * frand((unsigned)(o * 131 + g * 7 + 5)));
    }
  }
  for (int o = 0; o <= N; ++o) {
    for (int k = 0; k < K; ++k) {
      const unsigned qq = ((unsigned)(o * K + k) * 2654435761u >> 13) & 0xf;
      if (o < N) { q[(std::size_t)o * K + k] = (std::uint16_t)qq; }
      wq[(std::size_t)o * (K / 8) + k / 8] |=
          (std::uint32_t)qq << (4 * (k % 8));
    }
  }
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = (_Float16)(0.5f + frand((unsigned)(i * 7 + 3)));   // all non-zero
  }

  std::vector<float> ref((std::size_t)M * N);
  for (int m = 0; m < M; ++m) {
    for (int o = 0; o < N; ++o) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        const float s = (float)scales[(std::size_t)o * groups + k / G];
        const float b = (float)biases[(std::size_t)o * groups + k / G];
        acc += (double)(float)x[(std::size_t)m * K + k] *
               (s * (float)q[(std::size_t)o * K + k] + b);
      }
      ref[(std::size_t)m * N + o] = (float)acc;
    }
  }

  auto wbuf = mc->make_shared_buffer(wq.size() * 4);
  auto sbuf = mc->make_shared_buffer(scales.size() * 2);
  auto bbuf = mc->make_shared_buffer(biases.size() * 2);
  auto xbuf = mc->make_shared_buffer(x.size() * 2);
  auto ybuf = mc->make_shared_buffer((std::size_t)M * N * 2);
  std::memcpy(wbuf.contents(), wq.data(), wq.size() * 4);
  std::memcpy(sbuf.contents(), scales.data(), scales.size() * 2);
  std::memcpy(bbuf.contents(), biases.data(), biases.size() * 2);
  std::memcpy(xbuf.contents(), x.data(), x.size() * 2);
  auto stream = mc->make_command_stream();
  {
    auto enc = stream.begin_compute();
    enc.set_function(fn);
    enc.set_buffer(0, wbuf);
    enc.set_buffer(1, sbuf);
    enc.set_buffer(2, bbuf);
    enc.set_buffer(3, xbuf);
    enc.set_buffer(4, ybuf);
    enc.set_constant(5, K);
    enc.set_constant(6, N);
    enc.set_constant(7, M);
    enc.dispatch({32, (unsigned)(N / 4), (unsigned)((M + 1) / 2)}, {32, 2, 1});
  }
  stream.commit().wait();
  const auto* yp = static_cast<const _Float16*>(ybuf.contents());
  double maxrel = 0.0;
  for (int i = 0; i < M * N; ++i) {
    const double d = std::fabs((double)(float)yp[i] - (double)ref[i]);
    maxrel = std::fmax(maxrel, d / (std::fabs((double)ref[i]) + 1e-2));
  }
  std::printf("[qmv_batch_w4g32] K=%d max rel err = %.4g\n", K, maxrel);
  EXPECT_TRUE(maxrel < 0.03);
}

// Numerically verify the decode-only fused GeGLU GEMV at group-size 32
// (affine_qmv_geglu_w4g32 -- the GGUF gate/up decode kernel) against a CPU
// reference (interleaved gate/up rows, gelu_pytorch_tanh(gate)*up).
TEST(metal_lm_smoke, qmv_geglu_w4g32_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("affine_qmv");
  auto fn = lib.function("affine_qmv_geglu_w4g32");
  ASSERT_TRUE(fn.valid());

  // Tail guard for the fused gate/up GEMV: K = 3840 (Gemma-12B hidden) is NOT
  // a multiple of block_size 512. x is padded to the block boundary (Kpad) with
  // non-zero values and the weights carry an extra padding row, so without the
  // tail mask the partial last block folds in out-of-range data (amplified by
  // the gelu). Small biases -> isolates the tail (no affine cancellation).
  const int F = 2048, N = 2 * F, K = 3840, G = 32, groups = K / G;
  const int Kpad = ((K + 511) / 512) * 512;        // 4096
  std::vector<std::uint32_t> wq((std::size_t)(N + 1) * (K / 8), 0);
  std::vector<std::uint16_t> qn((std::size_t)N * K);
  std::vector<_Float16> scales((std::size_t)N * groups);
  std::vector<_Float16> biases((std::size_t)N * groups);
  std::vector<_Float16> x((std::size_t)Kpad);
  auto frand = [](unsigned i) {
    return (float)((i * 2654435761u) >> 9 & 0x3ff) / 1023.0f - 0.5f;
  };
  for (int o = 0; o < N; ++o) {
    for (int g = 0; g < groups; ++g) {
      scales[(std::size_t)o * groups + g] =
          (_Float16)(0.03f + 0.02f * frand((unsigned)(o * 91 + g)));
      biases[(std::size_t)o * groups + g] =
          (_Float16)(0.08f * frand((unsigned)(o * 31 + g * 7 + 5)));
    }
  }
  for (int o = 0; o <= N; ++o) {
    for (int k = 0; k < K; ++k) {
      const unsigned qq = ((unsigned)(o * K + k) * 2654435761u >> 13) & 0xf;
      if (o < N) { qn[(std::size_t)o * K + k] = (std::uint16_t)qq; }
      wq[(std::size_t)o * (K / 8) + k / 8] |=
          (std::uint32_t)qq << (4 * (k % 8));
    }
  }
  // Small positive activations (non-zero -> tail is exercised) kept tiny so the
  // squared geglu output gate*up stays inside fp16 range (gate ~ K*x*scale*q).
  for (int k = 0; k < Kpad; ++k) {
    x[k] = (_Float16)(0.04f + 0.06f * frand((unsigned)(k * 7 + 3)));
  }

  auto rowdot = [&](int o) {
    double acc = 0.0;
    for (int k = 0; k < K; ++k) {
      const float s = (float)scales[(std::size_t)o * groups + k / G];
      const float b = (float)biases[(std::size_t)o * groups + k / G];
      acc += (double)(float)x[k] * (s * (float)qn[(std::size_t)o * K + k] + b);
    }
    return (float)acc;
  };
  std::vector<float> ref((std::size_t)F);
  for (int g = 0; g < F; ++g) {
    const float gate = rowdot(2 * g), up = rowdot(2 * g + 1);
    const float t = std::tanh(0.7978845608028654f *
                              (gate + 0.044715f * gate * gate * gate));
    ref[g] = 0.5f * gate * (1.0f + t) * up;
  }

  auto wbuf = mc->make_shared_buffer(wq.size() * 4);
  auto sbuf = mc->make_shared_buffer(scales.size() * 2);
  auto bbuf = mc->make_shared_buffer(biases.size() * 2);
  auto xbuf = mc->make_shared_buffer(x.size() * 2);
  auto ybuf = mc->make_shared_buffer((std::size_t)F * 2);
  std::memcpy(wbuf.contents(), wq.data(), wq.size() * 4);
  std::memcpy(sbuf.contents(), scales.data(), scales.size() * 2);
  std::memcpy(bbuf.contents(), biases.data(), biases.size() * 2);
  std::memcpy(xbuf.contents(), x.data(), x.size() * 2);
  auto stream = mc->make_command_stream();
  {
    auto enc = stream.begin_compute();
    enc.set_function(fn);
    enc.set_buffer(0, wbuf);
    enc.set_buffer(1, sbuf);
    enc.set_buffer(2, bbuf);
    enc.set_buffer(3, xbuf);
    enc.set_buffer(4, ybuf);
    enc.set_constant(5, K);
    enc.set_constant(6, N);
    enc.dispatch({32, (unsigned)(F / 2), 1}, {32, 2, 1});
  }
  stream.commit().wait();
  const auto* yp = static_cast<const _Float16*>(ybuf.contents());
  double maxrel = 0.0;
  for (int g = 0; g < F; ++g) {
    const double d = std::fabs((double)(float)yp[g] - (double)ref[g]);
    maxrel = std::fmax(maxrel, d / (std::fabs((double)ref[g]) + 1e-2));
    std::printf("[geglu_w4g32] g=%2d gpu=%.5f ref=%.5f\n",
                g, (float)yp[g], ref[g]);
  }
  std::printf("[geglu_w4g32] max rel err = %.4g\n", maxrel);
  EXPECT_TRUE(maxrel < 0.05);
}
