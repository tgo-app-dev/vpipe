// moss.cc -- MOSS TTS: the delayed-pattern LM forward, the audio codec (decode
// / int8 / encode round trip / MMA-vs-scalar) and the end-to-end WAV smoke.

#include "tests/unit-tests/metal-lm/metal-lm-test-common.h"

// MOSS-TTS-8B (MossTTSDelay): the metal delay-pattern LM forward must agree
// with the mlx-audio reference. Full autoregressive token-exactness is
// IMPOSSIBLE here -- the audio heads are full of exact bf16 logit ties (two
// codes both at e.g. 55.5), so any tie-flip diverges + cascades, and the
// model README itself is non-reproducible across implementations. Instead we
// verify the FORWARD: teacher-force the reference rows and require that every
// early-window disagreement (before recurrent bf16 cache drift accumulates)
// is a numerical tie, and that the leading autoregressive rows are exact.
// Gated on a model dir + a golden-artifact dir produced by the mlx-audio
// reference (dump_golden.py): shapes.txt ("seq G channels"), input_ids.i32
// (the [seq, channels] prompt grid) and gen_delay.i32 (the [G, channels]
// reference generation, both raw little-endian int32 row-major).
TEST(metal_lm_smoke, moss_tts_delay_greedy_forward) {
  const char* path = std::getenv("VPIPE_MOSS_TTS_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS_TTS_GOLDEN");
  if (path == nullptr || *path == '\0' || gold == nullptr || *gold == '\0') {
    return;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::string gd(gold);
  int seq = 0, G = 0, ch = 0;
  {
    std::ifstream s(gd + "/shapes.txt");
    if (!s) { return; }
    s >> seq >> G >> ch;
  }
  ASSERT_TRUE(seq > 0 && G > 0 && ch == 33);
  auto read_i32 = [&](const std::string& fn, int count) {
    std::vector<std::int32_t> v((std::size_t)count, 0);
    std::ifstream f(gd + "/" + fn, std::ios::binary);
    if (f) {
      f.read(reinterpret_cast<char*>(v.data()),
             (std::streamsize)count * 4);
    }
    return v;
  };
  const std::vector<std::int32_t> iid = read_i32("input_ids.i32", seq * ch);
  const std::vector<std::int32_t> gen = read_i32("gen_delay.i32", G * ch);

  std::vector<std::vector<std::int32_t>> prompt(
      (std::size_t)seq, std::vector<std::int32_t>((std::size_t)ch, 0));
  for (int r = 0; r < seq; ++r) {
    for (int c = 0; c < ch; ++c) {
      prompt[(std::size_t)r][(std::size_t)c] =
          iid[(std::size_t)(r * ch + c)];
    }
  }

  auto model = genai::MetalMossTtsModel::load(path, mc);
  ASSERT_TRUE(model != nullptr && model->valid());

  // Reference generation rows [G][33].
  std::vector<std::vector<std::int32_t>> ref_rows(
      (std::size_t)G, std::vector<std::int32_t>((std::size_t)ch, 0));
  for (int r = 0; r < G; ++r) {
    for (int c = 0; c < ch; ++c) {
      ref_rows[(std::size_t)r][(std::size_t)c] = gen[(std::size_t)(r * ch + c)];
    }
  }

  // --- GATE: teacher-forced forward correctness ----------------------
  // The MOSS audio heads are full of EXACT bf16 logit ties (e.g. two codes
  // both at 55.5), so two bf16 implementations cannot agree on every greedy
  // argmax -- a single tie-flip then cascades. The forward is verified
  // correct if every active-codebook disagreement with the reference is a
  // numerical near-tie (my argmax's logit ~= the reference code's logit).
  const auto t0 = std::chrono::steady_clock::now();
  auto mism = model->teacher_force_audio_mismatches(prompt, ref_rows);
  const double dt = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  int active = 0;
  const int pad = model->config().audio_pad_code;
  for (int r = 0; r < G; ++r) {
    for (int cb = 0; cb < 32; ++cb) {
      if (gen[(std::size_t)(r * ch + 1 + cb)] != pad) { ++active; }
    }
  }
  constexpr float kTieTol = 0.75f;   // ~3 bf16 quanta at logit magnitude ~50
  constexpr int kEarly = 16;         // window before recurrent drift dominates
  int real_mismatch = 0, real_early = 0, first_real = -1, last_real = -1;
  float max_gap = 0.0f;
  int bucket[7] = {0, 0, 0, 0, 0, 0, 0};   // real mismatches per 10-row bucket
  for (const auto& m : mism) {
    const float gap = m.my_logit - m.ref_logit;   // >= 0 (my pick is argmax)
    if (gap > max_gap) { max_gap = gap; }
    if (gap > kTieTol) {
      ++real_mismatch;
      if (m.row < kEarly) { ++real_early; }
      if (first_real < 0) { first_real = m.row; }
      last_real = m.row;
      const int b = std::min(m.row / 10, 6);
      ++bucket[b];
    }
  }
  std::printf("[moss-tts-tf] %.2fs | active=%d disagreements=%zu "
              "(near-tie if gap<=%.2f) real(gap>tol)=%d max_gap=%.3f\n",
              dt, active, mism.size(), kTieTol, real_mismatch, max_gap);
  std::printf("[moss-tts-tf] real-mismatch rows: first=%d last=%d | per-10row "
              "buckets [0-9..60+]: %d %d %d %d %d %d %d | early(<%d)=%d\n",
              first_real, last_real, bucket[0], bucket[1], bucket[2], bucket[3],
              bucket[4], bucket[5], bucket[6], kEarly, real_early);
  for (std::size_t i = 0; i < mism.size() && i < 6; ++i) {
    std::printf("   row %d cb %d: mine=%d (%.3f) ref=%d (%.3f) gap=%.3f\n",
                mism[i].row, mism[i].codebook, mism[i].my_code,
                (double)mism[i].my_logit, mism[i].ref_code,
                (double)mism[i].ref_logit,
                (double)(mism[i].my_logit - mism[i].ref_logit));
  }
  // Forward correctness gate: BEFORE recurrent bf16 drift accumulates, every
  // disagreement must be a numerical tie (no gap>tol). Full-sequence
  // divergence is expected (the model README itself is non-reproducible
  // across implementations) and is judged by the decoded audio in Phase 2.
  EXPECT_TRUE(real_early == 0);

  // --- Smoke: the autoregressive loop + delay state machine. The leading
  // rows are exact (before any near-tie can flip); the run produces audio
  // and stops on its own. ---------------------------------------------
  auto outr = model->generate_delay_greedy(prompt, G);
  EXPECT_TRUE(!outr.empty());
  const int lead = std::min((int)outr.size(), std::min(G, 4));
  int lead_exact = 0;
  for (int r = 0; r < lead; ++r) {
    bool eq = true;
    for (int c = 0; c < ch; ++c) {
      if (outr[(std::size_t)r][(std::size_t)c] !=
          gen[(std::size_t)(r * ch + c)]) { eq = false; break; }
    }
    if (eq) { ++lead_exact; }
  }
  std::printf("[moss-tts] autoregressive gen=%d/%d | leading_exact_rows=%d/%d\n",
              (int)outr.size(), G, lead_exact, lead);
  EXPECT_TRUE(lead_exact == lead);
}

// MOSS Audio Tokenizer codec decode: the metal RVQ + 4-stage transformer
// decoder must reproduce the reference's per-stage tensors + final 24 kHz PCM
// within an f16 rel-L2 tolerance (the reference runs F32; we run f16). Gated
// on the codec dir + the golden dir (dump_codec_golden.py): codes_dedelay.i32
// + codes_shape.txt (the [T, n_vq] input), rvq.f32 / dec_stage0..7.f32 /
// wave.f32 (the reference intermediates, raw little-endian f32).
TEST(metal_lm_smoke, moss_codec_decode_rel_l2) {
  const char* path = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS_TTS_GOLDEN");
  if (path == nullptr || *path == '\0' || gold == nullptr || *gold == '\0') {
    return;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string gd(gold);

  int T = 0, nvq = 0;
  {
    std::ifstream s(gd + "/codes_shape.txt");
    if (!s) { return; }
    s >> T >> nvq;
  }
  ASSERT_TRUE(T > 0 && nvq == 32);
  std::vector<std::int32_t> flat((std::size_t)T * nvq, 0);
  {
    std::ifstream f(gd + "/codes_dedelay.i32", std::ios::binary);
    if (!f) { return; }
    f.read(reinterpret_cast<char*>(flat.data()), (std::streamsize)T * nvq * 4);
  }
  std::vector<std::vector<std::int32_t>> codes(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)nvq, 0));
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < nvq; ++c) {
      codes[(std::size_t)t][(std::size_t)c] = flat[(std::size_t)(t * nvq + c)];
    }
  }

  auto read_f32 = [&](const std::string& fn) {
    std::ifstream f(gd + "/" + fn, std::ios::binary | std::ios::ate);
    std::vector<float> v;
    if (!f) { return v; }
    const std::streamsize n = f.tellg() / 4;
    f.seekg(0);
    v.resize((std::size_t)n);
    f.read(reinterpret_cast<char*>(v.data()), n * 4);
    return v;
  };
  auto rel_l2 = [](const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) { return 9.99; }
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double d = (double)a[i] - (double)b[i];
      num += d * d;
      den += (double)b[i] * (double)b[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
  };

  auto codec = genai::MetalMossCodec::load(path, mc);
  ASSERT_TRUE(codec != nullptr && codec->valid());

  std::vector<std::vector<float>> stages;
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<float> wave = codec->decode(codes, &stages);
  const double dt = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(!wave.empty());

  const char* names[9] = {"rvq", "dec_stage0", "dec_stage1", "dec_stage2",
                          "dec_stage3", "dec_stage4", "dec_stage5",
                          "dec_stage6", "dec_stage7"};
  EXPECT_TRUE(stages.size() == 9);
  double worst = 0.0;
  for (std::size_t k = 0; k < stages.size() && k < 9; ++k) {
    const std::vector<float> ref = read_f32(std::string(names[k]) + ".f32");
    const double r = rel_l2(stages[k], ref);
    if (r > worst) { worst = r; }
    std::printf("[moss-codec] %-11s mine=%zu ref=%zu rel_l2=%.4f\n", names[k],
                stages[k].size(), ref.size(), r);
  }
  const std::vector<float> wref = read_f32("wave.f32");
  const double wr = rel_l2(wave, wref);
  std::printf("[moss-codec] %.2fs | wave samples=%zu/%zu rel_l2=%.4f | "
              "worst_stage=%.4f\n", dt, wave.size(), wref.size(), wr, worst);
  // f16 codec vs F32 reference: per-stage error stays small; the waveform is
  // the end-to-end arbiter.
  EXPECT_TRUE(wr < 0.05);
}

// Opt-in int8-g32 codec vs the f16 codec on the SAME (synthetic) codes -- the
// quantization-error quality gate. No LM / golden needed (the codec decode is
// deterministic), so this is fast (codec only). Reports rel-L2 of the int8
// waveform vs f16 and the resident-weight saving.
TEST(metal_lm_smoke, moss_codec_int8_rel_l2) {
  const char* path = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  if (path == nullptr || *path == '\0') { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  // Deterministic synthetic codes [T, 32] in [0, 1024).
  const int T = 48, nvq = 32;
  std::vector<std::vector<std::int32_t>> codes(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)nvq, 0));
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < nvq; ++c) {
      codes[(std::size_t)t][(std::size_t)c] = (t * 37 + c * 101 + 7) % 1024;
    }
  }

  auto f16 = genai::MetalMossCodec::load(path, mc, /*int8=*/false);
  ASSERT_TRUE(f16 != nullptr && f16->valid());
  const std::vector<float> w16 = f16->decode(codes, nullptr);
  f16.reset();
  ASSERT_TRUE(!w16.empty());

  auto i8 = genai::MetalMossCodec::load(path, mc, /*int8=*/true);
  ASSERT_TRUE(i8 != nullptr && i8->valid());
  const std::vector<float> w8 = i8->decode(codes, nullptr);
  ASSERT_TRUE(w8.size() == w16.size());

  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < w16.size(); ++i) {
    const double d = (double)w8[i] - (double)w16[i];
    num += d * d;
    den += (double)w16[i] * (double)w16[i];
  }
  const double rel_l2 = (den > 0.0) ? std::sqrt(num / den) : 0.0;
  std::printf("[moss-int8] samples=%zu rel_l2(int8 vs f16)=%.4f\n",
              w16.size(), rel_l2);
  // int8 g32 affine on the codec transformer GEMMs: a small waveform error.
  EXPECT_TRUE(rel_l2 < 0.10);
}

// Codec ENCODE (voice-cloning analysis path) round-trip. No torch oracle is
// available on the box, so correctness is checked oracle-free via the codec's
// fixed-point property: decode(C) lands exactly on code-set C's manifold, so a
// correct encoder recovers C from it -- encode(decode(C)) ~ C, and the iterate
// stabilizes. We start from deterministic codes C0, decode to an on-manifold
// waveform, then re-encode/re-decode twice and check (a) the waveform is
// reproduced (rel-L2 small + shrinking) and (b) the coarse codebooks converge
// to a fixed point. A broken encoder (wrong patch order / normalization /
// residual) yields garbage: rel-L2 ~1 and ~0 code agreement. Codec only (no
// LM, no golden); gated on VPIPE_MOSS_CODEC_MODEL.
TEST(metal_lm_smoke, moss_codec_encode_roundtrip) {
  const char* path = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  if (path == nullptr || *path == '\0') { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  auto codec = genai::MetalMossCodec::load(path, mc, /*int8=*/false,
                                           /*with_encoder=*/true);
  ASSERT_TRUE(codec != nullptr && codec->valid());
  ASSERT_TRUE(codec->has_encoder());

  const int T = 48, nvq = 32;
  std::vector<std::vector<std::int32_t>> c0(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)nvq, 0));
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < nvq; ++c) {
      c0[(std::size_t)t][(std::size_t)c] = (t * 37 + c * 101 + 7) % 1024;
    }
  }

  auto rel_l2 = [](const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) { return 1.0; }
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double d = (double)a[i] - (double)b[i];
      num += d * d;
      den += (double)b[i] * (double)b[i];
    }
    return (den > 0.0) ? std::sqrt(num / den) : 0.0;
  };
  // Per-codebook agreement of two code grids (fraction of frames equal).
  auto agree = [&](const std::vector<std::vector<std::int32_t>>& a,
                   const std::vector<std::vector<std::int32_t>>& b, int cb) {
    if (a.size() != b.size() || a.empty()) { return 0.0; }
    int n = 0;
    for (std::size_t t = 0; t < a.size(); ++t) {
      if (a[t][(std::size_t)cb] == b[t][(std::size_t)cb]) { ++n; }
    }
    return (double)n / (double)a.size();
  };

  const std::vector<float> w0 = codec->decode(c0, nullptr);
  ASSERT_TRUE(!w0.empty());
  const auto c1 = codec->encode(w0);
  ASSERT_TRUE(c1.size() == (std::size_t)T);
  const std::vector<float> w1 = codec->decode(c1, nullptr);
  const auto c2 = codec->encode(w1);
  ASSERT_TRUE(c2.size() == (std::size_t)T);
  const std::vector<float> w2 = codec->decode(c2, nullptr);

  const double rl_10 = rel_l2(w1, w0);
  const double rl_21 = rel_l2(w2, w1);
  std::printf("[moss-enc] T=%d  rel_l2(decode(enc(w0)),w0)=%.4f  "
              "rel_l2(2nd iterate)=%.4f\n", T, rl_10, rl_21);
  std::printf("[moss-enc] code agreement cb0/1/2/4/8  C1vsC0: "
              "%.2f %.2f %.2f %.2f %.2f\n",
              agree(c1, c0, 0), agree(c1, c0, 1), agree(c1, c0, 2),
              agree(c1, c0, 4), agree(c1, c0, 8));
  std::printf("[moss-enc] code agreement cb0/1/2/4/8  C2vsC1: "
              "%.2f %.2f %.2f %.2f %.2f\n",
              agree(c2, c1, 0), agree(c2, c1, 1), agree(c2, c1, 2),
              agree(c2, c1, 4), agree(c2, c1, 8));

  // Waveform is reproduced (not garbage) and the iterate stabilizes (the
  // second round-trip is no worse than the first). The fixed-point signal:
  // the coarsest codebook converges (C2 ~ C1 far better than C1 ~ C0).
  ASSERT_TRUE(w1.size() == w0.size() && w2.size() == w1.size());
  EXPECT_TRUE(std::isfinite(rl_10) && rl_10 < 0.6);
  EXPECT_TRUE(std::isfinite(rl_21) && rl_21 <= rl_10 + 0.05);
  EXPECT_TRUE(agree(c2, c1, 0) > 0.7);
}

// Codec decode throughput: f16 vs int8-g32 (warm, same codes). Reports the
// per-decode ms and the int8/f16 ratio so the int8 dequant overhead is
// visible. Gated on VPIPE_MOSS_CODEC_BENCH (codec only; fast).
TEST(metal_lm_smoke, moss_codec_bench) {
  const char* path = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  if (path == nullptr || *path == '\0' ||
      std::getenv("VPIPE_MOSS_CODEC_BENCH") == nullptr) {
    return;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  int T = 48;
  if (const char* e = std::getenv("VPIPE_MOSS_CODEC_BENCH_T")) {
    T = std::max(1, std::atoi(e));
  }
  const int nvq = 32;
  std::vector<std::vector<std::int32_t>> codes(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)nvq, 0));
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < nvq; ++c) {
      codes[(std::size_t)t][(std::size_t)c] = (t * 37 + c * 101 + 7) % 1024;
    }
  }
  using Clk = std::chrono::steady_clock;
  auto bench = [&](bool int8) -> double {
    auto codec = genai::MetalMossCodec::load(path, mc, int8);
    if (codec == nullptr || !codec->valid()) { return -1.0; }
    for (int i = 0; i < 3; ++i) { (void)codec->decode(codes, nullptr); }  // warm
    const int K = 10;
    const auto t0 = Clk::now();
    for (int i = 0; i < K; ++i) { (void)codec->decode(codes, nullptr); }
    return std::chrono::duration<double, std::milli>(Clk::now() - t0).count()
           / K;
  };
  const double f16 = bench(false);
  const double i8 = bench(true);
  std::printf("[moss-codec-bench] T=%d  f16=%.1f ms  int8=%.1f ms  "
              "(int8/f16=%.2fx, +%.1f ms)\n",
              T, f16, i8, (f16 > 0 ? i8 / f16 : 0.0), i8 - f16);
  EXPECT_TRUE(f16 > 0 && i8 > 0);
}

// M5 matrix-core decode paths (matmul2d f16 GEMM + windowed-causal flash
// attention) must be numerically equivalent to the steel/scalar path: decode
// the same codes with each mma lever on vs off and compare the waveform
// rel-L2. Skips if the model env is unset or the mma paths are not active.
TEST(metal_lm_smoke, moss_codec_mma_matches_scalar) {
  const char* path = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  if (path == nullptr || *path == '\0') { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto codec = genai::MetalMossCodec::load(path, mc, /*int8=*/false);
  ASSERT_TRUE(codec != nullptr && codec->valid());
  if (!codec->use_attn_mma() && !codec->use_mma2()) {
    std::fprintf(stderr, "[moss-codec] mma paths not active (pre-M5?), skip\n");
    return;
  }
  const int T = 53, nvq = codec->n_quantizers();   // non-16-multiple: tail
  std::vector<std::vector<std::int32_t>> codes(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)nvq, 0));
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < nvq; ++c) {
      codes[(std::size_t)t][(std::size_t)c] = (t * 37 + c * 101 + 7) % 1024;
    }
  }
  auto rl2 = [](const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double d = (double)a[i] - (double)b[i];
      num += d * d; den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  };
  codec->set_use_mma2(true);  codec->set_use_attn_mma(true);
  const std::vector<float> both = codec->decode(codes, nullptr);
  ASSERT_TRUE(!both.empty());
  codec->set_use_attn_mma(false);                 // scalar attn, mma gemm
  const std::vector<float> no_attn = codec->decode(codes, nullptr);
  codec->set_use_attn_mma(true); codec->set_use_mma2(false);  // steel gemm
  const std::vector<float> no_gemm = codec->decode(codes, nullptr);
  codec->set_use_mma2(true);
  ASSERT_TRUE(no_attn.size() == both.size() && no_gemm.size() == both.size());
  const double r_attn = rl2(both, no_attn);
  const double r_gemm = rl2(both, no_gemm);
  std::fprintf(stderr, "[moss-codec] attn mma-vs-scalar rel-L2=%.6f | "
               "gemm mma-vs-steel rel-L2=%.6f\n", r_attn, r_gemm);
  EXPECT_TRUE(r_attn < 1e-2);
  EXPECT_TRUE(r_gemm < 1e-2);
}

// The int8 (w8 g32) matrix-core GEMM path (dequant-once via affine_dequant_
// w8g32 -> the f16 dense_gemm_mma) must match the fused affine STEEL w8 GEMM:
// decode the same codes with the int8 mma GEMM on vs off (attention held on
// mma) and compare the waveform rel-L2. Skips if env unset or mma inactive.
TEST(metal_lm_smoke, moss_codec_int8_mma_matches_steel) {
  const char* path = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  if (path == nullptr || *path == '\0') { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto codec = genai::MetalMossCodec::load(path, mc, /*int8=*/true);
  ASSERT_TRUE(codec != nullptr && codec->valid());
  if (!codec->use_mma2()) {
    std::fprintf(stderr, "[moss-codec] int8 mma not active (pre-M5?), skip\n");
    return;
  }
  const int T = 53, nvq = codec->n_quantizers();
  std::vector<std::vector<std::int32_t>> codes(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)nvq, 0));
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < nvq; ++c) {
      codes[(std::size_t)t][(std::size_t)c] = (t * 37 + c * 101 + 7) % 1024;
    }
  }
  codec->set_use_mma2(true);
  const std::vector<float> a = codec->decode(codes, nullptr);
  codec->set_use_mma2(false);
  const std::vector<float> b = codec->decode(codes, nullptr);
  codec->set_use_mma2(true);
  ASSERT_TRUE(!a.empty() && a.size() == b.size());
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d; den += (double)b[i] * (double)b[i];
  }
  const double rl2 = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  std::fprintf(stderr, "[moss-codec] int8 GEMM mma-vs-steel rel-L2 = %.6f\n",
               rl2);
  EXPECT_TRUE(rl2 < 1e-2);
}

// End-to-end metal MOSS-TTS: LM (delay-pattern code generation) -> de-delay +
// drop all-pad frames (the reference _decode_generated_audio pipeline) ->
// codec -> 24 kHz PCM, written to a playable WAV. Gated on the LM + codec
// dirs + the golden dir (for the prompt grid). Writes
// $VPIPE_MOSS_TTS_GOLDEN/e2e.wav.
TEST(metal_lm_smoke, moss_tts_end_to_end_wav) {
  const char* lm_path = std::getenv("VPIPE_MOSS_TTS_MODEL");
  const char* cc_path = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS_TTS_GOLDEN");
  if (!lm_path || !*lm_path || !cc_path || !*cc_path || !gold || !*gold) {
    return;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string gd(gold);

  int seq = 0, G = 0, ch = 0;
  {
    std::ifstream s(gd + "/shapes.txt");
    if (!s) { return; }
    s >> seq >> G >> ch;
  }
  ASSERT_TRUE(seq > 0 && ch == 33);
  std::vector<std::int32_t> iid((std::size_t)seq * ch, 0);
  {
    std::ifstream f(gd + "/input_ids.i32", std::ios::binary);
    if (!f) { return; }
    f.read(reinterpret_cast<char*>(iid.data()), (std::streamsize)seq * ch * 4);
  }
  std::vector<std::vector<std::int32_t>> prompt(
      (std::size_t)seq, std::vector<std::int32_t>((std::size_t)ch, 0));
  for (int r = 0; r < seq; ++r) {
    for (int c = 0; c < ch; ++c) {
      prompt[(std::size_t)r][(std::size_t)c] = iid[(std::size_t)(r * ch + c)];
    }
  }

  const int n_vq = 32, pad = 1024;
  std::vector<std::vector<std::int32_t>> gen;
  {
    auto lm = genai::MetalMossTtsModel::load(lm_path, mc);
    ASSERT_TRUE(lm != nullptr && lm->valid());
    gen = lm->generate_delay_greedy(prompt, 1024);   // [Gg][33]
  }  // free the 8B LM before loading the codec (16 GB box)
  ASSERT_TRUE(!gen.empty());
  const int Gg = (int)gen.size();

  // De-delay: tokens[t][cb] = audio_codes[cb + t][cb], audio_codes = gen[:,1:].
  const int out_len = Gg - n_vq + 1;
  std::vector<std::vector<std::int32_t>> codes;
  for (int t = 0; t < out_len; ++t) {
    std::vector<std::int32_t> row((std::size_t)n_vq, 0);
    bool all_pad = true;
    for (int cb = 0; cb < n_vq; ++cb) {
      int v = gen[(std::size_t)(cb + t)][(std::size_t)(1 + cb)];
      if (v != pad) { all_pad = false; }
      if (v < 0 || v >= pad) { v = pad - 1; }   // clamp pad/OOB to a valid code
      row[(std::size_t)cb] = v;
    }
    if (!all_pad) { codes.push_back(std::move(row)); }   // drop all-pad frames
  }
  std::printf("[moss-e2e] LM gen rows=%d -> de-delay %d -> non-pad frames=%zu\n",
              Gg, out_len, codes.size());
  ASSERT_TRUE(!codes.empty());

  std::vector<float> wave;
  int sr = 24000;
  {
    auto codec = genai::MetalMossCodec::load(cc_path, mc);
    ASSERT_TRUE(codec != nullptr && codec->valid());
    sr = codec->sample_rate();
    wave = codec->decode(codes, nullptr);
  }
  ASSERT_TRUE(!wave.empty());
  double peak = 0.0;
  for (float s : wave) { peak = std::max(peak, (double)std::fabs(s)); }
  std::printf("[moss-e2e] codec -> %zu samples = %.2fs @ %dHz | peak=%.3f\n",
              wave.size(), wave.size() / (double)sr, sr, peak);
  EXPECT_TRUE(peak > 0.01);   // produced actual audio, not silence

  // Write a 16-bit PCM mono WAV.
  const std::string wav = gd + "/e2e.wav";
  std::ofstream out(wav, std::ios::binary);
  const std::uint32_t n = (std::uint32_t)wave.size();
  const std::uint32_t data_bytes = n * 2;
  const std::uint32_t byte_rate = (std::uint32_t)sr * 2;
  auto u32 = [&](std::uint32_t v) { out.write(reinterpret_cast<char*>(&v), 4); };
  auto u16 = [&](std::uint16_t v) { out.write(reinterpret_cast<char*>(&v), 2); };
  out.write("RIFF", 4); u32(36 + data_bytes); out.write("WAVE", 4);
  out.write("fmt ", 4); u32(16); u16(1); u16(1);
  u32((std::uint32_t)sr); u32(byte_rate); u16(2); u16(16);
  out.write("data", 4); u32(data_bytes);
  for (float s : wave) {
    int v = (int)std::lround(std::max(-1.0f, std::min(1.0f, s)) * 32767.0f);
    u16((std::uint16_t)(std::int16_t)v);
  }
  out.close();
  std::printf("[moss-e2e] wrote %s\n", wav.c_str());
}
