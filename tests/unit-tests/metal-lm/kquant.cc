// kquant.cc -- GGUF k-quant kernels: Q4_K / Q5_K / Q6_K dequant and matvec
// against CPU references, their bandwidth / rps sweeps, and the k-quant ->
// affine repack (correctness + bench).

#include "tests/unit-tests/metal-lm/metal-lm-test-common.h"

// Native Q6_K (llama.cpp k-quant) GPU unpack must match the CPU dequant
// bit-for-bit (it's a lossless format, not a requant). Synthesizes raw Q6_K
// super-blocks and compares dequant_q6k_f16 against an inline CPU reference
// mirroring gguf-file.cc's dequant_row_q6_K.
TEST(metal_lm_smoke, q6k_dequant_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  auto fn = lib.function("dequant_q6k_f16");
  ASSERT_TRUE(fn.valid());

  const int nsb = 3, N = nsb * 256;
  std::vector<std::uint8_t> blk((std::size_t)nsb * 210);
  for (int sb = 0; sb < nsb; ++sb) {
    std::uint8_t* p = blk.data() + (std::size_t)sb * 210;
    for (int i = 0; i < 128; ++i) {                 // ql
      p[i] = (std::uint8_t)((sb * 128 + i) * 37 + 11);
    }
    for (int i = 0; i < 64; ++i) {                  // qh
      p[128 + i] = (std::uint8_t)((sb * 64 + i) * 53 + 7);
    }
    for (int i = 0; i < 16; ++i) {                  // int8 scales
      p[192 + i] = (std::uint8_t)(std::int8_t)((sb * 16 + i) * 5 - 40);
    }
    const _Float16 d = (_Float16)(0.005f + 0.001f * sb);
    std::memcpy(p + 208, &d, 2);
  }

  // CPU reference (mirrors gguf-file.cc kQ6_K).
  std::vector<float> ref((std::size_t)N);
  for (int sb = 0; sb < nsb; ++sb) {
    const std::uint8_t* p = blk.data() + (std::size_t)sb * 210;
    const std::uint8_t* ql = p;
    const std::uint8_t* qh = p + 128;
    const auto* sc = reinterpret_cast<const std::int8_t*>(p + 192);
    _Float16 d16;
    std::memcpy(&d16, p + 208, 2);
    const float d = (float)d16;
    float* y = ref.data() + (std::size_t)sb * 256;
    for (int half = 0; half < 2; ++half) {
      const int qlo = half * 64, qho = half * 32, sco = half * 8, yo = half * 128;
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16, hi = qh[qho + l];
        const int q1 = ((ql[qlo + l] & 0xF) | (((hi >> 0) & 3) << 4)) - 32;
        const int q2 = ((ql[qlo + l + 32] & 0xF) | (((hi >> 2) & 3) << 4)) - 32;
        const int q3 = ((ql[qlo + l] >> 4) | (((hi >> 4) & 3) << 4)) - 32;
        const int q4 = ((ql[qlo + l + 32] >> 4) | (((hi >> 6) & 3) << 4)) - 32;
        y[yo + l] = d * sc[sco + is + 0] * q1;
        y[yo + l + 32] = d * sc[sco + is + 2] * q2;
        y[yo + l + 64] = d * sc[sco + is + 4] * q3;
        y[yo + l + 96] = d * sc[sco + is + 6] * q4;
      }
    }
  }

  auto sbuf = mc->make_shared_buffer(blk.size());
  auto obuf = mc->make_shared_buffer((std::size_t)N * 2);
  std::memcpy(sbuf.contents(), blk.data(), blk.size());
  auto stream = mc->make_command_stream();
  {
    auto enc = stream.begin_compute();
    enc.set_function(fn);
    enc.set_buffer(0, sbuf);
    enc.set_buffer(1, obuf);
    enc.set_constant(2, N);
    enc.dispatch({(unsigned)N, 1, 1}, {256, 1, 1});
  }
  stream.commit().wait();
  const auto* op = static_cast<const _Float16*>(obuf.contents());
  double maxrel = 0.0;
  for (int i = 0; i < N; ++i) {
    const double d = std::fabs((double)(float)op[i] - (double)ref[i]);
    maxrel = std::fmax(maxrel, d / (std::fabs((double)ref[i]) + 1e-3));
  }
  std::printf("[q6k_dequant] N=%d max rel err = %.4g\n", N, maxrel);
  EXPECT_TRUE(maxrel < 1e-2);   // f16 output rounding only
}

// Native Q6_K lm_head GEMV (qmv_q6k_f16): y[o] = sum_h x[h]*dequant(W[o,h]),
// verified against a CPU dequant+dot reference. This is the lossless,
// memory-saving replacement for the 8-bit affine-requant lm_head GEMV.
TEST(metal_lm_smoke, qmv_q6k_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  auto fn = lib.function("qmv_q6k_f16");
  ASSERT_TRUE(fn.valid());

  const int N = 64, H = 512, sbpr = H / 256;        // super-blocks per row
  std::vector<std::uint8_t> w((std::size_t)N * sbpr * 210);
  std::vector<_Float16> x((std::size_t)H);
  auto frand = [](unsigned i) {
    return (float)((i * 2654435761u) >> 9 & 0x3ff) / 1023.0f - 0.5f;
  };
  for (int o = 0; o < N; ++o) {
    for (int s = 0; s < sbpr; ++s) {
      std::uint8_t* p = w.data() + ((std::size_t)o * sbpr + s) * 210;
      const unsigned base = (unsigned)(o * sbpr + s);
      for (int i = 0; i < 128; ++i) { p[i] = (std::uint8_t)((base * 128 + i) * 37 + 11); }
      for (int i = 0; i < 64; ++i) { p[128 + i] = (std::uint8_t)((base * 64 + i) * 53 + 7); }
      for (int i = 0; i < 16; ++i) {
        p[192 + i] = (std::uint8_t)(std::int8_t)((base * 16 + i) * 5 - 40);
      }
      const _Float16 d = (_Float16)(0.004f + 0.0005f * (float)(base % 8));
      std::memcpy(p + 208, &d, 2);
    }
  }
  for (int h = 0; h < H; ++h) { x[h] = (_Float16)frand((unsigned)(h * 7 + 3)); }

  // CPU reference: dequant each weight (mirror gguf-file kQ6_K) and dot with x.
  auto q6k_cpu = [&](const std::uint8_t* sb, int pos) {
    const std::uint8_t* ql = sb;
    const std::uint8_t* qh = sb + 128;
    const auto* sc = reinterpret_cast<const std::int8_t*>(sb + 192);
    _Float16 d16; std::memcpy(&d16, sb + 208, 2);
    const int hf = pos >> 7, p = pos & 127, which = p >> 5, l = p & 31;
    const int is = l >> 4, qlo = hf * 64, qho = hf * 32, sco = hf * 8;
    const int hi = qh[qho + l];
    int q, sci;
    if (which == 0) { q = (ql[qlo + l] & 0xF) | (((hi >> 0) & 3) << 4); sci = sco + is; }
    else if (which == 1) { q = (ql[qlo + l + 32] & 0xF) | (((hi >> 2) & 3) << 4); sci = sco + is + 2; }
    else if (which == 2) { q = (ql[qlo + l] >> 4) | (((hi >> 4) & 3) << 4); sci = sco + is + 4; }
    else { q = (ql[qlo + l + 32] >> 4) | (((hi >> 6) & 3) << 4); sci = sco + is + 6; }
    return (float)d16 * (float)sc[sci] * (float)(q - 32);
  };
  std::vector<float> ref((std::size_t)N);
  for (int o = 0; o < N; ++o) {
    double acc = 0.0;
    for (int h = 0; h < H; ++h) {
      const std::uint8_t* sb = w.data() + ((std::size_t)o * sbpr + h / 256) * 210;
      acc += (double)(float)x[h] * (double)q6k_cpu(sb, h & 255);
    }
    ref[o] = (float)acc;
  }

  auto wbuf = mc->make_shared_buffer(w.size());
  auto xbuf = mc->make_shared_buffer(x.size() * 2);
  auto ybuf = mc->make_shared_buffer((std::size_t)N * 2);
  std::memcpy(wbuf.contents(), w.data(), w.size());
  std::memcpy(xbuf.contents(), x.data(), x.size() * 2);
  // Both the original and the llama.cpp-style v2 kernel must match the CPU
  // reference (v2 is not bit-identical to the original -- different fp grouping
  // -- but must be numerically equivalent).
  auto run = [&](const char* name) {
    auto f = lib.function(name);
    if (!f.valid()) { return; }
    auto stream = mc->make_command_stream();
    {
      auto enc = stream.begin_compute();
      enc.set_function(f);
      enc.set_buffer(0, wbuf);
      enc.set_buffer(1, xbuf);
      enc.set_buffer(2, ybuf);
      enc.set_constant(3, H);
      enc.set_constant(4, N);
      enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
    }
    stream.commit().wait();
    const auto* yp = static_cast<const _Float16*>(ybuf.contents());
    double maxrel = 0.0;
    for (int o = 0; o < N; ++o) {
      const double d = std::fabs((double)(float)yp[o] - (double)ref[o]);
      maxrel = std::fmax(maxrel, d / (std::fabs((double)ref[o]) + 1e-2));
    }
    std::printf("[qmv_q6k] %-14s N=%d H=%d max rel err = %.4g\n",
                name, N, H, maxrel);
    EXPECT_TRUE(maxrel < 0.02);
  };
  run("qmv_q6k_f16");
  run("qmv_q6k_v2_f16");
}

// Shared 6-bit scale/min unpack for the Q4_K/Q5_K CPU references below
// (mirrors gguf-file.cc get_scale_min_k4_ / llama.cpp get_scale_min_k4).
namespace {
inline void gsmk4_cpu(int j, const std::uint8_t* q, std::uint8_t& d,
                      std::uint8_t& m) {
  if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
  else { d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
         m = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4); }
}
}  // namespace

// Native Q4_K dequant (dequant_q4k_f16) vs a CPU reference mirroring
// gguf-file.cc kQ4_K. 144-byte super-block: d(f16) dmin(f16) scales[12] qs[128].
TEST(metal_lm_smoke, q4k_dequant_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  auto fn = lib.function("dequant_q4k_f16");
  ASSERT_TRUE(fn.valid());

  const int nsb = 3, N = nsb * 256;
  std::vector<std::uint8_t> blk((std::size_t)nsb * 144);
  for (int sb = 0; sb < nsb; ++sb) {
    std::uint8_t* p = blk.data() + (std::size_t)sb * 144;
    const _Float16 d = (_Float16)(0.005f + 0.001f * sb);
    const _Float16 dm = (_Float16)(0.05f + 0.01f * sb);
    std::memcpy(p, &d, 2); std::memcpy(p + 2, &dm, 2);
    for (int i = 0; i < 12; ++i) { p[4 + i] = (std::uint8_t)((sb*12+i)*29+3); }
    for (int i = 0; i < 128; ++i) { p[16 + i] = (std::uint8_t)((sb*128+i)*37+11); }
  }
  std::vector<float> ref((std::size_t)N);
  for (int sb = 0; sb < nsb; ++sb) {
    const std::uint8_t* p = blk.data() + (std::size_t)sb * 144;
    _Float16 d16, m16; std::memcpy(&d16, p, 2); std::memcpy(&m16, p + 2, 2);
    const float d = (float)d16, dmin = (float)m16;
    const std::uint8_t* scales = p + 4; const std::uint8_t* qs = p + 16;
    float* y = ref.data() + (std::size_t)sb * 256;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
      std::uint8_t sc, m;
      gsmk4_cpu(is + 0, scales, sc, m); const float d1 = d*sc, m1 = dmin*m;
      gsmk4_cpu(is + 1, scales, sc, m); const float d2 = d*sc, m2 = dmin*m;
      const std::uint8_t* q = qs + (j / 64) * 32;
      for (int l = 0; l < 32; ++l) {
        y[j + l]      = d1 * (q[l] & 0x0F) - m1;
        y[j + l + 32] = d2 * (q[l] >> 4)  - m2;
      }
      is += 2;
    }
  }
  auto sbuf = mc->make_shared_buffer(blk.size());
  auto obuf = mc->make_shared_buffer((std::size_t)N * 2);
  std::memcpy(sbuf.contents(), blk.data(), blk.size());
  auto stream = mc->make_command_stream();
  { auto enc = stream.begin_compute();
    enc.set_function(fn); enc.set_buffer(0, sbuf); enc.set_buffer(1, obuf);
    enc.set_constant(2, N); enc.dispatch({(unsigned)N, 1, 1}, {256, 1, 1}); }
  stream.commit().wait();
  const auto* op = static_cast<const _Float16*>(obuf.contents());
  double maxrel = 0.0;
  for (int i = 0; i < N; ++i) {
    const double dd = std::fabs((double)(float)op[i] - (double)ref[i]);
    maxrel = std::fmax(maxrel, dd / (std::fabs((double)ref[i]) + 1e-3));
  }
  std::printf("[q4k_dequant] N=%d max rel err = %.4g\n", N, maxrel);
  EXPECT_TRUE(maxrel < 1e-2);
}

// Native Q5_K dequant (dequant_q5k_f16) vs CPU reference mirroring kQ5_K.
// 176-byte super-block: d(f16) dmin(f16) scales[12] qh[32] qs[128].
TEST(metal_lm_smoke, q5k_dequant_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  auto fn = lib.function("dequant_q5k_f16");
  ASSERT_TRUE(fn.valid());

  const int nsb = 3, N = nsb * 256;
  std::vector<std::uint8_t> blk((std::size_t)nsb * 176);
  for (int sb = 0; sb < nsb; ++sb) {
    std::uint8_t* p = blk.data() + (std::size_t)sb * 176;
    const _Float16 d = (_Float16)(0.005f + 0.001f * sb);
    const _Float16 dm = (_Float16)(0.05f + 0.01f * sb);
    std::memcpy(p, &d, 2); std::memcpy(p + 2, &dm, 2);
    for (int i = 0; i < 12; ++i) { p[4 + i] = (std::uint8_t)((sb*12+i)*29+3); }
    for (int i = 0; i < 32; ++i) { p[16 + i] = (std::uint8_t)((sb*32+i)*43+5); }
    for (int i = 0; i < 128; ++i) { p[48 + i] = (std::uint8_t)((sb*128+i)*37+11); }
  }
  std::vector<float> ref((std::size_t)N);
  for (int sb = 0; sb < nsb; ++sb) {
    const std::uint8_t* p = blk.data() + (std::size_t)sb * 176;
    _Float16 d16, m16; std::memcpy(&d16, p, 2); std::memcpy(&m16, p + 2, 2);
    const float d = (float)d16, dmin = (float)m16;
    const std::uint8_t* scales = p + 4;
    const std::uint8_t* qh = p + 16; const std::uint8_t* qs = p + 48;
    float* y = ref.data() + (std::size_t)sb * 256;
    int is = 0; std::uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
      std::uint8_t sc, m;
      gsmk4_cpu(is + 0, scales, sc, m); const float d1 = d*sc, m1 = dmin*m;
      gsmk4_cpu(is + 1, scales, sc, m); const float d2 = d*sc, m2 = dmin*m;
      const std::uint8_t* q = qs + (j / 64) * 32;
      for (int l = 0; l < 32; ++l) {
        const int lo = (q[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
        const int hi = (q[l] >> 4)   + ((qh[l] & u2) ? 16 : 0);
        y[j + l]      = d1 * lo - m1;
        y[j + l + 32] = d2 * hi - m2;
      }
      is += 2; u1 <<= 2; u2 <<= 2;
    }
  }
  auto sbuf = mc->make_shared_buffer(blk.size());
  auto obuf = mc->make_shared_buffer((std::size_t)N * 2);
  std::memcpy(sbuf.contents(), blk.data(), blk.size());
  auto stream = mc->make_command_stream();
  { auto enc = stream.begin_compute();
    enc.set_function(fn); enc.set_buffer(0, sbuf); enc.set_buffer(1, obuf);
    enc.set_constant(2, N); enc.dispatch({(unsigned)N, 1, 1}, {256, 1, 1}); }
  stream.commit().wait();
  const auto* op = static_cast<const _Float16*>(obuf.contents());
  double maxrel = 0.0;
  for (int i = 0; i < N; ++i) {
    const double dd = std::fabs((double)(float)op[i] - (double)ref[i]);
    maxrel = std::fmax(maxrel, dd / (std::fabs((double)ref[i]) + 1e-3));
  }
  std::printf("[q5k_dequant] N=%d max rel err = %.4g\n", N, maxrel);
  EXPECT_TRUE(maxrel < 1e-2);
}

// Native Q4_K / Q5_K GEMV (qmv_q4k_f16 / qmv_q5k_f16): y[o]=sum_h x[h]*W[o,h],
// verified against a CPU dequant+dot reference. These back the GGUF Qwen3.5
// linears' decode path (no affine requant).
TEST(metal_lm_smoke, qmv_q4k_q5k_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  ASSERT_TRUE(lib.function("qmv_q4k_f16").valid());
  ASSERT_TRUE(lib.function("qmv_q5k_f16").valid());

  const int N = 64, H = 512, sbpr = H / 256;
  auto frand = [](unsigned i) {
    return (float)((i * 2654435761u) >> 9 & 0x3ff) / 1023.0f - 0.5f;
  };
  std::vector<_Float16> x((std::size_t)H);
  for (int h = 0; h < H; ++h) { x[h] = (_Float16)frand((unsigned)(h*7+3)); }

  auto fill_block = [&](std::uint8_t* p, unsigned base, int blk_bytes,
                        int qh_off, int qs_off) {
    const _Float16 d = (_Float16)(0.004f + 0.0005f * (float)(base % 8));
    const _Float16 dm = (_Float16)(0.03f + 0.002f * (float)(base % 5));
    std::memcpy(p, &d, 2); std::memcpy(p + 2, &dm, 2);
    for (int i = 0; i < 12; ++i) { p[4+i] = (std::uint8_t)((base*12+i)*29+3); }
    if (qh_off >= 0) {
      for (int i = 0; i < 32; ++i) {
        p[qh_off+i] = (std::uint8_t)((base*32+i)*43+5);
      }
    }
    for (int i = 0; i < 128; ++i) {
      p[qs_off+i] = (std::uint8_t)((base*128+i)*37+11);
    }
    (void)blk_bytes;
  };

  // --- Q4_K ---
  {
    std::vector<std::uint8_t> w((std::size_t)N * sbpr * 144);
    for (int o = 0; o < N; ++o) {
      for (int s = 0; s < sbpr; ++s) {
        fill_block(w.data() + ((std::size_t)o*sbpr+s)*144,
                   (unsigned)(o*sbpr+s), 144, -1, 16);
      }
    }
    auto q4k_cpu = [&](const std::uint8_t* sb, int pos) {
      _Float16 d16, m16; std::memcpy(&d16, sb, 2); std::memcpy(&m16, sb+2, 2);
      const float d = (float)d16, dmin = (float)m16;
      const std::uint8_t* scales = sb + 4; const std::uint8_t* qs = sb + 16;
      const int chunk = pos >> 6, within = pos & 63;
      const int is = chunk*2 + (within >> 5), l = within & 31;
      const unsigned qb = qs[chunk*32 + l];
      const unsigned nib = (within < 32) ? (qb & 0x0F) : (qb >> 4);
      std::uint8_t sc, m; gsmk4_cpu(is, scales, sc, m);
      return d * sc * (float)nib - dmin * m;
    };
    std::vector<float> ref((std::size_t)N);
    for (int o = 0; o < N; ++o) {
      double acc = 0.0;
      for (int h = 0; h < H; ++h) {
        const std::uint8_t* sb = w.data() + ((std::size_t)o*sbpr + h/256)*144;
        acc += (double)(float)x[h] * (double)q4k_cpu(sb, h & 255);
      }
      ref[o] = (float)acc;
    }
    auto wbuf = mc->make_shared_buffer(w.size());
    auto xbuf = mc->make_shared_buffer(x.size() * 2);
    auto ybuf = mc->make_shared_buffer((std::size_t)N * 2);
    std::memcpy(wbuf.contents(), w.data(), w.size());
    std::memcpy(xbuf.contents(), x.data(), x.size() * 2);
    auto stream = mc->make_command_stream();
    { auto enc = stream.begin_compute();
      enc.set_function(lib.function("qmv_q4k_f16"));
      enc.set_buffer(0, wbuf); enc.set_buffer(1, xbuf); enc.set_buffer(2, ybuf);
      enc.set_constant(3, H); enc.set_constant(4, N);
      enc.dispatch({32, (unsigned)(((N + 1) / 2) * 2), 1}, {32, 2, 1}); }
    stream.commit().wait();
    const auto* yp = static_cast<const _Float16*>(ybuf.contents());
    double maxrel = 0.0;
    for (int o = 0; o < N; ++o) {
      const double dd = std::fabs((double)(float)yp[o] - (double)ref[o]);
      maxrel = std::fmax(maxrel, dd / (std::fabs((double)ref[o]) + 1e-2));
    }
    std::printf("[qmv_q4k] N=%d H=%d max rel err = %.4g\n", N, H, maxrel);
    EXPECT_TRUE(maxrel < 0.02);
  }
  // --- Q5_K ---
  {
    std::vector<std::uint8_t> w((std::size_t)N * sbpr * 176);
    for (int o = 0; o < N; ++o) {
      for (int s = 0; s < sbpr; ++s) {
        fill_block(w.data() + ((std::size_t)o*sbpr+s)*176,
                   (unsigned)(o*sbpr+s), 176, 16, 48);
      }
    }
    auto q5k_cpu = [&](const std::uint8_t* sb, int pos) {
      _Float16 d16, m16; std::memcpy(&d16, sb, 2); std::memcpy(&m16, sb+2, 2);
      const float d = (float)d16, dmin = (float)m16;
      const std::uint8_t* scales = sb + 4;
      const std::uint8_t* qh = sb + 16; const std::uint8_t* qs = sb + 48;
      const int chunk = pos >> 6, within = pos & 63;
      const int is = chunk*2 + (within >> 5), l = within & 31;
      const unsigned qb = qs[chunk*32 + l];
      unsigned nib = (within < 32) ? (qb & 0x0F) : (qb >> 4);
      const int bit = 2*chunk + ((within < 32) ? 0 : 1);
      nib += ((unsigned(qh[l]) >> bit) & 1u) * 16u;
      std::uint8_t sc, m; gsmk4_cpu(is, scales, sc, m);
      return d * sc * (float)nib - dmin * m;
    };
    std::vector<float> ref((std::size_t)N);
    for (int o = 0; o < N; ++o) {
      double acc = 0.0;
      for (int h = 0; h < H; ++h) {
        const std::uint8_t* sb = w.data() + ((std::size_t)o*sbpr + h/256)*176;
        acc += (double)(float)x[h] * (double)q5k_cpu(sb, h & 255);
      }
      ref[o] = (float)acc;
    }
    auto wbuf = mc->make_shared_buffer(w.size());
    auto xbuf = mc->make_shared_buffer(x.size() * 2);
    auto ybuf = mc->make_shared_buffer((std::size_t)N * 2);
    std::memcpy(wbuf.contents(), w.data(), w.size());
    std::memcpy(xbuf.contents(), x.data(), x.size() * 2);
    auto stream = mc->make_command_stream();
    { auto enc = stream.begin_compute();
      enc.set_function(lib.function("qmv_q5k_f16"));
      enc.set_buffer(0, wbuf); enc.set_buffer(1, xbuf); enc.set_buffer(2, ybuf);
      enc.set_constant(3, H); enc.set_constant(4, N);
      enc.dispatch({32, (unsigned)(((N + 1) / 2) * 2), 1}, {32, 2, 1}); }
    stream.commit().wait();
    const auto* yp = static_cast<const _Float16*>(ybuf.contents());
    double maxrel = 0.0;
    for (int o = 0; o < N; ++o) {
      const double dd = std::fabs((double)(float)yp[o] - (double)ref[o]);
      maxrel = std::fmax(maxrel, dd / (std::fabs((double)ref[o]) + 1e-2));
    }
    std::printf("[qmv_q5k] N=%d H=%d max rel err = %.4g\n", N, H, maxrel);
    EXPECT_TRUE(maxrel < 0.02);
  }
}

// Q6_K lm_head GEMV bandwidth at the real 12B shape [vocab=262144, hidden=3840]
// -- settles whether qmv_q6k_f16 is DRAM-bound (already at the ~100 GB/s M4
// ceiling, so llama.cpp's "read each ql/qh byte once, extract all nibbles
// in-thread" trick can't help) or load/instruction-bound (the per-nibble
// re-reads across lanes cost, and the trick would). Reports GB/s over the raw
// Q6_K table bytes (210 B / 256 wt). Gated on VPIPE_Q6K_BW.
TEST(metal_lm_smoke, q6k_lmhead_bandwidth) {
  if (std::getenv("VPIPE_Q6K_BW") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");

  const int N = 262144, H = 3840, sbpr = H / 256;     // 15 super-blocks/row
  const std::size_t wbytes = (std::size_t)N * sbpr * 210;
  auto wbuf = mc->make_shared_buffer(wbytes);
  auto xbuf = mc->make_shared_buffer((std::size_t)H * 2);
  auto ybuf = mc->make_shared_buffer((std::size_t)N * 2);
  // Content is irrelevant for timing; just fill x with something finite.
  auto* xp = static_cast<_Float16*>(xbuf.contents());
  for (int h = 0; h < H; ++h) { xp[h] = (_Float16)0.01f; }

  auto fn1 = lib.function("qmv_q6k_f16");
  auto fn2 = lib.function("qmv_q6k_v2_f16");
  auto once = [&](metal_compute::ComputeFunction& fn) {
    auto st = mc->make_command_stream();
    { auto enc = st.begin_compute();
      enc.set_function(fn);
      enc.set_buffer(0, wbuf); enc.set_buffer(1, xbuf); enc.set_buffer(2, ybuf);
      enc.set_constant(3, H); enc.set_constant(4, N);
      enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
    }
    st.commit().wait();
  };
  auto measure = [&](metal_compute::ComputeFunction& fn) {
    const int R = 20;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < R; ++i) { once(fn); }
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count() / R;
  };
  // Interleave v1/v2 measures and take the min (peak-clock) of each to cancel
  // the GPU's per-run clock drift.
  for (int w = 0; w < 5; ++w) { once(fn1); once(fn2); }   // warm
  double m1 = 1e18, m2 = 1e18;
  for (int k = 0; k < 6; ++k) {
    m1 = std::fmin(m1, measure(fn1));
    m2 = std::fmin(m2, measure(fn2));
  }
  std::printf("[q6k_bw] qmv_q6k_f16    %.3f ms  %.1f GB/s (min-of-6)\n",
              m1, (double)wbytes / (m1 * 1e6));
  std::printf("[q6k_bw] qmv_q6k_v2_f16 %.3f ms  %.1f GB/s (min-of-6)  %.2fx\n",
              m2, (double)wbytes / (m2 * 1e6), m1 / m2);
  EXPECT_TRUE(true);
}

// K-quant qmv achieved bandwidth on the 27B FFN gate/up shape [N=17408,
// H=5120] (the dominant decode weight read). Contrasts q4k/q5k (old single-row
// kernel) against q6k_v2 (4-row) and reports GB/s of k-quant bytes moved, to
// locate the decode bandwidth gap vs the ~240 GB/s affine qmv. Gated on
// VPIPE_KQUANT_BW.
TEST(metal_lm_smoke, qmv_kquant_bandwidth) {
  if (std::getenv("VPIPE_KQUANT_BW") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  // Big N so the kernel saturates and per-command-buffer submit overhead is
  // amortized (matches q6k_lmhead_bandwidth's methodology); H = 27B FFN width.
  const int N = 262144, H = 5120, sbpr = H / 256;    // 20 super-blocks/row
  auto xbuf = mc->make_shared_buffer((std::size_t)H * 2);
  auto ybuf = mc->make_shared_buffer((std::size_t)N * 2);
  auto* xp = static_cast<_Float16*>(xbuf.contents());
  for (int h = 0; h < H; ++h) { xp[h] = (_Float16)0.01f; }
  // (kernel, block bytes, rows-per-threadgroup) -- dispatch matches each
  // kernel's documented grid (NSG=2 simdgroups/tg; q6k_v2 does RPS=4 rows).
  struct K { const char* name; int blk; int rpt; };
  const K ks[] = {{"qmv_q4k_f16", 144, 2}, {"qmv_q5k_f16", 176, 2},
                  {"qmv_q6k_f16", 210, 2}, {"qmv_q6k_v2_f16", 210, 8}};
  for (const K& k : ks) {
    auto fn = lib.function(k.name);
    if (!fn.valid()) { std::printf("[kq_bw] %-14s missing\n", k.name); continue; }
    const std::size_t wbytes = (std::size_t)N * sbpr * k.blk;
    auto wbuf = mc->make_shared_buffer(wbytes);
    const unsigned gy = (unsigned)(((N + k.rpt - 1) / k.rpt) * 2);
    auto once = [&]() {
      auto st = mc->make_command_stream();
      { auto enc = st.begin_compute();
        enc.set_function(fn);
        enc.set_buffer(0, wbuf); enc.set_buffer(1, xbuf); enc.set_buffer(2, ybuf);
        enc.set_constant(3, H); enc.set_constant(4, N);
        enc.dispatch({32, gy, 1}, {32, 2, 1});
      }
      st.commit().wait();
    };
    for (int w = 0; w < 5; ++w) { once(); }
    double m = 1e18;
    for (int t = 0; t < 6; ++t) {
      const auto t0 = std::chrono::steady_clock::now();
      const int R = 20;
      for (int i = 0; i < R; ++i) { once(); }
      m = std::fmin(m, std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count() / R);
    }
    std::printf("[kq_bw] %-14s %6.1f MB  %.3f ms  %6.1f GB/s\n", k.name,
                (double)wbytes / 1e6, m, (double)wbytes / (m * 1e6));
  }
  EXPECT_TRUE(true);
}

// Sweep qmv_q6k_v2 RPS/NSG tuning on the real 27B Q6_K shapes (lm_head, FFN
// down, GDN qkv). The production default is <4,2>; llama.cpp uses nr0=2 + a
// device-tuned nsg. Gated on VPIPE_Q6K_SWEEP.
TEST(metal_lm_smoke, qmv_q6k_v2_rps_sweep) {
  if (std::getenv("VPIPE_Q6K_SWEEP") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  struct V { const char* name; int rps; int nsg; };
  const V vars[] = {{"qmv_q6k_v2_f16", 4, 2}, {"qmv_q6k_v2_r2n2_f16", 2, 2},
                    {"qmv_q6k_v2_r8n2_f16", 8, 2}, {"qmv_q6k_v2_r2n4_f16", 2, 4},
                    {"qmv_q6k_v2_r4n4_f16", 4, 4}, {"qmv_q6k_v2_r1n4_f16", 1, 4},
                    {"qmv_q6k_v2_r1n8_f16", 1, 8}, {"qmv_q6k_v2_r2n8_f16", 2, 8},
                    {"qmv_q6k_v2_r4n8_f16", 4, 8}};
  struct S { const char* name; int N; int H; };
  const S shapes[] = {{"lm_head ", 248320, 5120}, {"ffn_down", 5120, 17408},
                      {"gdn_qkv ", 10240, 5120}};
  for (const S& sh : shapes) {
    const int nsb = sh.H / 256;
    const std::size_t wbytes = (std::size_t)sh.N * nsb * 210;
    auto wbuf = mc->make_shared_buffer(wbytes);
    auto xbuf = mc->make_shared_buffer((std::size_t)sh.H * 2);
    auto ybuf = mc->make_shared_buffer((std::size_t)sh.N * 2);
    auto* xp = static_cast<_Float16*>(xbuf.contents());
    for (int h = 0; h < sh.H; ++h) { xp[h] = (_Float16)0.01f; }
    double best_ms = 1e18; const char* best = "";
    for (const V& v : vars) {
      auto fn = lib.function(v.name);
      if (!fn.valid()) { std::printf("  %s MISSING\n", v.name); continue; }
      const unsigned rpt = (unsigned)(v.rps * v.nsg);
      const unsigned gy = ((sh.N + rpt - 1) / rpt) * (unsigned)v.nsg;
      auto run = [&](int R) {
        auto st = mc->make_command_stream();
        { auto e = st.begin_compute(); e.set_function(fn);
          e.set_buffer(0, wbuf); e.set_buffer(1, xbuf); e.set_buffer(2, ybuf);
          e.set_constant(3, sh.H); e.set_constant(4, sh.N);
          for (int r = 0; r < R; ++r) {
            e.dispatch({32u, gy, 1u}, {32u, (unsigned)v.nsg, 1u});
          } }
        st.commit().wait();
      };
      run(5);
      double m = 1e18;
      for (int t = 0; t < 5; ++t) {
        const auto t0 = std::chrono::steady_clock::now();
        run(40);
        m = std::fmin(m, std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count() / 40);
      }
      std::printf("[q6k_sweep %s] %-22s r%dn%d  %.3f ms  %6.1f GB/s\n",
                  sh.name, v.name, v.rps, v.nsg, m, wbytes / (m * 1e6));
      if (m < best_ms) { best_ms = m; best = v.name; }
    }
    std::printf("[q6k_sweep %s] BEST = %s (%.1f GB/s)\n", sh.name, best,
                wbytes / (best_ms * 1e6));
  }
  EXPECT_TRUE(true);
}

// Validates + benchmarks the Q4_K -> affine-g32 load-time repack: repacks a
// random Q4_K matrix and checks affine_qmv_w4g32 over it matches the native
// qmv_q4k (lossless dequant), then times both on the 27B FFN shape. Confirms
// the bit-layout + the decode bandwidth win before the loader integration.
// Gated on VPIPE_Q4K_AFFINE_BW.
TEST(metal_lm_smoke, q4k_affine_repack_matches_and_bench) {
  if (std::getenv("VPIPE_Q4K_AFFINE_BW") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto elt = mc->load_library("llm_elementwise");
  auto aff = mc->load_library("affine_qmv");
  auto f_repack = elt.function("repack_q4k_to_affine_g32");
  auto f_q4k    = elt.function("qmv_q4k_f16");
  auto f_aff    = aff.function("affine_qmv_w4g32");
  ASSERT_TRUE(f_repack.valid() && f_q4k.valid() && f_aff.valid());

  const int N = 4096, H = 5120, nsb = H / 256;
  auto src = mc->make_shared_buffer((std::size_t)N * nsb * 144);
  std::mt19937 rng(123);
  std::uniform_int_distribution<int> ub(0, 255);
  auto* sp = static_cast<std::uint8_t*>(src.contents());
  for (std::size_t i = 0; i < (std::size_t)N * nsb * 144; ++i) {
    sp[i] = (std::uint8_t)ub(rng);
  }
  auto x = mc->make_shared_buffer((std::size_t)H * 2);
  auto* xp = static_cast<_Float16*>(x.contents());
  for (int h = 0; h < H; ++h) { xp[h] = (_Float16)((ub(rng) - 128) * 0.001f); }
  auto wq    = mc->make_shared_buffer((std::size_t)N * (H / 8) * 4);
  auto sc    = mc->make_shared_buffer((std::size_t)N * (H / 32) * 2);
  auto bs    = mc->make_shared_buffer((std::size_t)N * (H / 32) * 2);
  auto y_aff = mc->make_shared_buffer((std::size_t)N * 2);
  auto y_kq  = mc->make_shared_buffer((std::size_t)N * 2);

  auto submit = [&](auto setup) {
    auto st = mc->make_command_stream();
    { auto e = st.begin_compute(); setup(e); }
    st.commit().wait();
  };
  submit([&](metal_compute::ComputeEncoder& e) {
    e.set_function(f_repack);
    e.set_buffer(0, src); e.set_buffer(1, wq); e.set_buffer(2, sc);
    e.set_buffer(3, bs); e.set_constant(4, H); e.set_constant(5, N);
    e.dispatch({(unsigned)nsb, (unsigned)N, 1}, {(unsigned)nsb, 32, 1});
  });
  submit([&](metal_compute::ComputeEncoder& e) {
    e.set_function(f_aff);
    e.set_buffer(0, wq); e.set_buffer(1, sc); e.set_buffer(2, bs);
    e.set_buffer(3, x); e.set_buffer(4, y_aff);
    e.set_constant(5, H); e.set_constant(6, N);
    e.dispatch({32u, (unsigned)(N / 4), 1u}, {32u, 2u, 1u});
  });
  submit([&](metal_compute::ComputeEncoder& e) {
    e.set_function(f_q4k);
    e.set_buffer(0, src); e.set_buffer(1, x); e.set_buffer(2, y_kq);
    e.set_constant(3, H); e.set_constant(4, N);
    e.dispatch({32u, (unsigned)(((N + 1) / 2) * 2), 1u}, {32u, 2u, 1u});
  });
  const auto* ya = static_cast<const _Float16*>(y_aff.contents());
  const auto* yk = static_cast<const _Float16*>(y_kq.contents());
  double maxrel = 0.0;
  for (int o = 0; o < N; ++o) {
    const double a = (double)(float)ya[o], k = (double)(float)yk[o];
    maxrel = std::max(maxrel, std::fabs(a - k) / std::max(1e-4, std::fabs(k)));
  }
  std::printf("[q4k_affine] N=%d H=%d  max rel err (affine vs q4k) = %.4g\n",
              N, H, maxrel);
  EXPECT_TRUE(maxrel < 0.02);

  auto bench = [&](bool affine) -> double {
    const int R = 200;
    auto run = [&]() {
      auto st = mc->make_command_stream();
      { auto e = st.begin_compute();
        e.set_function(affine ? f_aff : f_q4k);
        if (affine) {
          e.set_buffer(0, wq); e.set_buffer(1, sc); e.set_buffer(2, bs);
          e.set_buffer(3, x); e.set_buffer(4, y_aff);
          e.set_constant(5, H); e.set_constant(6, N);
        } else {
          e.set_buffer(0, src); e.set_buffer(1, x); e.set_buffer(2, y_kq);
          e.set_constant(3, H); e.set_constant(4, N);
        }
        for (int r = 0; r < R; ++r) {
          if (affine) { e.dispatch({32u, (unsigned)(N / 4), 1u}, {32u, 2u, 1u}); }
          else { e.dispatch({32u, (unsigned)(((N + 1) / 2) * 2), 1u},
                            {32u, 2u, 1u}); }
        } }
      st.commit().wait();
    };
    for (int w = 0; w < 3; ++w) { run(); }
    double best = 1e18;
    for (int t = 0; t < 5; ++t) {
      const auto t0 = std::chrono::steady_clock::now();
      run();
      best = std::fmin(best, std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count() / R);
    }
    return best;
  };
  const double q4k_bytes = (double)N * nsb * 144;
  const double aff_bytes = (double)N * ((H / 8) * 4 + 2 * (H / 32) * 2);
  const double t_kq = bench(false), t_af = bench(true);
  std::printf("[q4k_affine] qmv_q4k       %.3f ms  %.1f GB/s\n",
              t_kq, q4k_bytes / (t_kq * 1e6));
  std::printf("[q4k_affine] affine_w4g32  %.3f ms  %.1f GB/s  (%.2fx faster)\n",
              t_af, aff_bytes / (t_af * 1e6), t_kq / t_af);
  EXPECT_TRUE(true);
}

// Q5_K -> affine-8bit-g32 experiment: validates the repack (vs qmv_q5k) and
// times affine_qmv_w8g32 against the native qmv_q5k on a GDN-out shape. The
// 8-bit affine reads MORE bytes (9 vs 5.5 bits/wt) -- this measures whether the
// higher kernel bandwidth still nets a win. Gated on VPIPE_Q5K_AFFINE_BW.
TEST(metal_lm_smoke, q5k_affine_repack_matches_and_bench) {
  if (std::getenv("VPIPE_Q5K_AFFINE_BW") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto elt = mc->load_library("llm_elementwise");
  auto aff = mc->load_library("affine_qmv");
  auto f_repack = elt.function("repack_q5k_to_affine_g32");
  auto f_q5k    = elt.function("qmv_q5k_f16");
  auto f_aff    = aff.function("affine_qmv_w8g32");
  ASSERT_TRUE(f_repack.valid() && f_q5k.valid() && f_aff.valid());

  const int N = 6144, H = 5120, nsb = H / 256;       // ~GDN out_proj shape
  auto src = mc->make_shared_buffer((std::size_t)N * nsb * 176);   // Q5_K block
  std::mt19937 rng(321);
  std::uniform_int_distribution<int> ub(0, 255);
  auto* sp = static_cast<std::uint8_t*>(src.contents());
  for (std::size_t i = 0; i < (std::size_t)N * nsb * 176; ++i) {
    sp[i] = (std::uint8_t)ub(rng);
  }
  auto x = mc->make_shared_buffer((std::size_t)H * 2);
  auto* xp = static_cast<_Float16*>(x.contents());
  for (int h = 0; h < H; ++h) { xp[h] = (_Float16)((ub(rng) - 128) * 0.001f); }
  auto wq    = mc->make_shared_buffer((std::size_t)N * H);          // 8-bit
  auto sc    = mc->make_shared_buffer((std::size_t)N * (H / 32) * 2);
  auto bs    = mc->make_shared_buffer((std::size_t)N * (H / 32) * 2);
  auto y_aff = mc->make_shared_buffer((std::size_t)N * 2);
  auto y_kq  = mc->make_shared_buffer((std::size_t)N * 2);

  auto submit = [&](auto setup) {
    auto st = mc->make_command_stream();
    { auto e = st.begin_compute(); setup(e); }
    st.commit().wait();
  };
  submit([&](metal_compute::ComputeEncoder& e) {
    e.set_function(f_repack);
    e.set_buffer(0, src); e.set_buffer(1, wq); e.set_buffer(2, sc);
    e.set_buffer(3, bs); e.set_constant(4, H); e.set_constant(5, N);
    e.dispatch({(unsigned)nsb, (unsigned)N, 1}, {(unsigned)nsb, 1, 1});
  });
  submit([&](metal_compute::ComputeEncoder& e) {
    e.set_function(f_aff);
    e.set_buffer(0, wq); e.set_buffer(1, sc); e.set_buffer(2, bs);
    e.set_buffer(3, x); e.set_buffer(4, y_aff);
    e.set_constant(5, H); e.set_constant(6, N);
    e.dispatch({32u, (unsigned)(N / 4), 1u}, {32u, 2u, 1u});
  });
  submit([&](metal_compute::ComputeEncoder& e) {
    e.set_function(f_q5k);
    e.set_buffer(0, src); e.set_buffer(1, x); e.set_buffer(2, y_kq);
    e.set_constant(3, H); e.set_constant(4, N);
    e.dispatch({32u, (unsigned)(((N + 1) / 2) * 2), 1u}, {32u, 2u, 1u});
  });
  const auto* ya = static_cast<const _Float16*>(y_aff.contents());
  const auto* yk = static_cast<const _Float16*>(y_kq.contents());
  double maxrel = 0.0;
  for (int o = 0; o < N; ++o) {
    const double a = (double)(float)ya[o], k = (double)(float)yk[o];
    maxrel = std::max(maxrel, std::fabs(a - k) / std::max(1e-4, std::fabs(k)));
  }
  std::printf("[q5k_affine] N=%d H=%d  max rel err (affine vs q5k) = %.4g\n",
              N, H, maxrel);
  EXPECT_TRUE(maxrel < 0.02);

  auto bench = [&](bool affine) -> double {
    const int R = 200;
    auto run = [&]() {
      auto st = mc->make_command_stream();
      { auto e = st.begin_compute();
        e.set_function(affine ? f_aff : f_q5k);
        if (affine) {
          e.set_buffer(0, wq); e.set_buffer(1, sc); e.set_buffer(2, bs);
          e.set_buffer(3, x); e.set_buffer(4, y_aff);
          e.set_constant(5, H); e.set_constant(6, N);
        } else {
          e.set_buffer(0, src); e.set_buffer(1, x); e.set_buffer(2, y_kq);
          e.set_constant(3, H); e.set_constant(4, N);
        }
        for (int r = 0; r < R; ++r) {
          if (affine) { e.dispatch({32u, (unsigned)(N / 4), 1u}, {32u, 2u, 1u}); }
          else { e.dispatch({32u, (unsigned)(((N + 1) / 2) * 2), 1u},
                            {32u, 2u, 1u}); }
        } }
      st.commit().wait();
    };
    for (int w = 0; w < 3; ++w) { run(); }
    double best = 1e18;
    for (int t = 0; t < 5; ++t) {
      const auto t0 = std::chrono::steady_clock::now();
      run();
      best = std::fmin(best, std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count() / R);
    }
    return best;
  };
  const double q5k_bytes = (double)N * nsb * 176;
  const double aff_bytes = (double)N * (H + 2 * (H / 32) * 2);   // 8-bit + s/b
  const double t_kq = bench(false), t_af = bench(true);
  std::printf("[q5k_affine] qmv_q5k       %.3f ms  %.1f GB/s (%.1f MB)\n",
              t_kq, q5k_bytes / (t_kq * 1e6), q5k_bytes / 1e6);
  std::printf("[q5k_affine] affine_w8g32  %.3f ms  %.1f GB/s (%.1f MB)  "
              "(%.2fx vs q5k time)\n", t_af, aff_bytes / (t_af * 1e6),
              aff_bytes / 1e6, t_kq / t_af);
  EXPECT_TRUE(true);
}
