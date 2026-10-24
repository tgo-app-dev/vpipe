// Synthetic round-trip for the forward affine quantizer + safetensors writer
// + the MetalLlamaWeights reader. Proves three things WITHOUT a real model:
//   1. quantize_linear -> affine_dequant reproduces the source within the
//      expected per-group quantization error (the quant math).
//   2. write -> MetalLlamaWeights::open_model -> load -> dequant is
//      BYTE-IDENTICAL to the in-memory dequant (the writer<->reader contract).
//   3. mixed-precision per-tensor bits are inferred correctly from the file.
//
// Always-on (no env gate): it allocates tiny GPU buffers and needs only a
// working metal-compute runtime; skips vacuously if none is present.

#include "minitest.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/quantize/affine-quantizer.h"
#include "generative-models/quantize/safetensors-writer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using namespace vpipe::metal_compute;

namespace {

namespace fs = std::filesystem;

// Dequant a packed affine weight (in the affine_dequant metallib f16 form)
// into a fresh f16 [N,K] buffer. bits in {4,8}, group 64.
SharedBuffer
dequant_(MetalCompute* mc, const ComputeLibrary& lib, const SharedBuffer& w,
         const SharedBuffer& s, const SharedBuffer& b, int N, int K, int bits)
{
  ComputeFunction fn =
      lib.function(bits == 8 ? "affine_dequant_w8g64" : "affine_dequant_w4g64");
  SharedBuffer y = mc->make_shared_buffer((std::size_t)N * K * 2);
  if (!fn.valid() || y.empty()) { return {}; }
  CommandStream stream = mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    enc.set_function(fn);
    enc.set_buffer(0, w);
    enc.set_buffer(1, s);
    enc.set_buffer(2, b);
    enc.set_buffer(3, y);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    enc.dispatch({(unsigned)((std::size_t)K * bits / 32), (unsigned)N, 1},
                 {64, 1, 1});
  }
  stream.commit().wait();
  return y;
}

double
rel_l2_(const _Float16* a, const _Float16* b, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double x = (double)a[i], y = (double)b[i];
    num += (x - y) * (x - y);
    den += x * x;
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace

TEST(model_quantize_roundtrip, affine_writer_reader_dequant)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }                 // no metal runtime -> skip
  ComputeLibrary deq = mc->load_library("affine_dequant");
  if (!deq.valid()) { return; }
  AffineQuantizer q(mc);
  ASSERT_TRUE(q.valid());

  const int N = 96, K = 256, group = 64;         // 4 groups/row, 96 rows
  // Synthetic source weights: a smooth per-row ramp + a small ripple, in a
  // realistic [-0.6, 0.6]-ish range, written as f16.
  SharedBuffer w_in = mc->make_shared_buffer((std::size_t)N * K * 2);
  ASSERT_TRUE(!w_in.empty());
  auto* src = static_cast<_Float16*>(w_in.contents());
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      const double v = 0.4 * std::sin(0.05 * k + 0.3 * n) +
                       0.1 * std::cos(0.7 * k) + 0.002 * (k - K / 2);
      src[(std::size_t)n * K + k] = (_Float16)v;
    }
  }

  const fs::path tmp = fs::temp_directory_path() /
                       ("vpipe-quant-rt-" + std::to_string(::getpid()));

  for (int bits : {8, 4}) {
    SharedBuffer w_out, s_out, b_out;
    ASSERT_TRUE(q.quantize_linear(w_in, /*src_bf16=*/false, N, K, bits, group,
                                  /*clip=*/1.0f, w_out, s_out, b_out));
    ASSERT_TRUE(!w_out.empty() && !s_out.empty() && !b_out.empty());
    ASSERT_TRUE(w_out.byte_size() == AffineQuantizer::weight_bytes(N, K, bits));
    ASSERT_TRUE(s_out.byte_size() == AffineQuantizer::scale_bytes(N, K, group));

    // In-memory dequant (ground truth for the contract check).
    SharedBuffer deq_direct = dequant_(mc, deq, w_out, s_out, b_out, N, K, bits);
    ASSERT_TRUE(!deq_direct.empty());

    // Write the three tensors, then read them back through the real reader.
    const fs::path dir = tmp / ("b" + std::to_string(bits));
    {
      SafetensorsWriter wr(dir.string());
      const std::int64_t wcols = (std::int64_t)K * bits / 32;
      const std::int64_t gcols = (std::int64_t)K / group;
      ASSERT_TRUE(wr.add("t.weight", "U32", {N, wcols}, w_out.contents(),
                         w_out.byte_size()));
      ASSERT_TRUE(wr.add("t.scales", "F16", {N, gcols}, s_out.contents(),
                         s_out.byte_size()));
      ASSERT_TRUE(wr.add("t.biases", "F16", {N, gcols}, b_out.contents(),
                         b_out.byte_size()));
      ASSERT_TRUE(wr.close());
    }
    auto wts = MetalLlamaWeights::open_model(dir.string());
    ASSERT_TRUE(wts.has_value());
    ASSERT_TRUE(wts->has("t.weight") && wts->has("t.scales") &&
                wts->has("t.biases"));
    SharedBuffer fw = wts->load("t.weight", mc);
    SharedBuffer fs_ = wts->load("t.scales", mc);
    SharedBuffer fb = wts->load("t.biases", mc);
    ASSERT_TRUE(!fw.empty() && !fs_.empty() && !fb.empty());
    ASSERT_TRUE(fw.byte_size() == w_out.byte_size());

    SharedBuffer deq_file = dequant_(mc, deq, fw, fs_, fb, N, K, bits);
    ASSERT_TRUE(!deq_file.empty());

    // (2) writer<->reader fidelity: file path == in-memory path, exactly.
    const auto* dd = static_cast<const _Float16*>(deq_direct.contents());
    const auto* df = static_cast<const _Float16*>(deq_file.contents());
    const double contract = rel_l2_(dd, df, (std::size_t)N * K);
    EXPECT_TRUE(contract == 0.0);

    // (1) quant math: dequant tracks the source within per-group error.
    const double err = rel_l2_(dd, src, (std::size_t)N * K);
    const double tol = (bits == 8) ? 0.03 : 0.20;
    EXPECT_TRUE(err < tol);
  }

  std::error_code ec;
  fs::remove_all(tmp, ec);
}

// THE RUN DTYPE (QuantizeOptions::run_dtype, AffineQuantizer::Scales): a
// pack meant for a bf16 run writes BF16 scales and biases, one for an f16
// run F16 ones, and either fits its codes to that ROUNDED pair; the
// original format writes F16 fit to the unrounded scale. For each width,
// group and source dtype, and each format:
//   - the codes are exactly clamp(round((w - b) / s)) on the pair as
//     stored, restated on the CPU (the fitted formats) -- a code at a
//     rounding tie may sit either side of it, the GPU's reciprocal being
//     an ulp from the CPU's;
//   - dequantized as the model that runs it holds the pair -- a bf16 model
//     rounds an F16 pair to bf16 at load -- the fitted pack is no further
//     from the source than the original one, for its own run;
//   - the BF16 pair survives the safetensors writer and reader by dtype;
//   - for w8 g64 from bf16 the bf16 fit is BYTE-IDENTICAL to the in-memory
//     w8 build's kernel (affine_quant_rows_w8g64), which model-quantize
//     uses there.
TEST(model_quantize_roundtrip, a_run_dtype_fits_its_scales)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  AffineQuantizer q(mc);
  ASSERT_TRUE(q.valid());
  using Sc = AffineQuantizer::Scales;
  auto bf16_bits = [](float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (std::uint16_t)(u >> 16);
  };
  auto from_bf16 = [](std::uint16_t h) {
    const std::uint32_t u = (std::uint32_t)h << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, 4);
    return f;
  };
  auto from_f16 = [](std::uint16_t h) {
    _Float16 v;
    std::memcpy(&v, &h, 2);
    return (float)v;
  };
  const int N = 128, K = 512;
  const fs::path tmp = fs::temp_directory_path() /
                       ("vpipe-quant-rd-" + std::to_string(::getpid()));
  for (const bool src_bf16 : {true, false}) {
    // Weights spread as a DiT's are, with a few wide groups.
    std::vector<float> wf((std::size_t)N * K);
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) {
        double v = 0.02 * std::sin(0.37 * k + 1.3 * n) +
                   0.01 * std::cos(0.11 * k * (n % 7 + 1));
        if ((k * 13 + n) % 97 == 0) { v *= 9.0; }
        wf[(std::size_t)n * K + k] = (float)v;
      }
    }
    SharedBuffer w_in = mc->make_shared_buffer((std::size_t)N * K * 2);
    ASSERT_TRUE(!w_in.empty());
    if (w_in.empty()) { continue; }
    auto* raw = static_cast<std::uint16_t*>(w_in.contents());
    for (std::size_t i = 0; i < wf.size(); ++i) {
      if (src_bf16) {
        raw[i] = bf16_bits(wf[i]);
        wf[i] = from_bf16(raw[i]);
      } else {
        const _Float16 h = (_Float16)wf[i];
        std::memcpy(&raw[i], &h, 2);
        wf[i] = (float)h;
      }
    }
    for (const int bits : {8, 4}) {
      for (const int group : {64, 32}) {
        const int G = K / group, VPW = 32 / bits, QMAX = (1 << bits) - 1;
        // The pack's values as a model of `run_bf16` holds its pair.
        auto values = [&](const SharedBuffer& w, const SharedBuffer& sc,
                          const SharedBuffer& bi, bool pair_bf16,
                          bool run_bf16) {
          const auto* wp = static_cast<const std::uint32_t*>(w.contents());
          const auto* sp = static_cast<const std::uint16_t*>(sc.contents());
          const auto* bp = static_cast<const std::uint16_t*>(bi.contents());
          std::vector<float> out((std::size_t)N * K);
          for (int n = 0; n < N; ++n) {
            for (int g = 0; g < G; ++g) {
              float s1 = pair_bf16 ? from_bf16(sp[n * G + g])
                                   : from_f16(sp[n * G + g]);
              float b1 = pair_bf16 ? from_bf16(bp[n * G + g])
                                   : from_f16(bp[n * G + g]);
              if (run_bf16 && !pair_bf16) {
                s1 = from_bf16(bf16_bits(s1));
                b1 = from_bf16(bf16_bits(b1));
              }
              for (int i = 0; i < group; ++i) {
                const int k = g * group + i;
                const std::uint32_t word =
                    wp[(std::size_t)n * (K / VPW) + k / VPW];
                const int code = (int)((word >> (bits * (k % VPW))) &
                                       (std::uint32_t)QMAX);
                out[(std::size_t)n * K + k] = (float)code * s1 + b1;
              }
            }
          }
          return out;
        };
        auto err = [&](const std::vector<float>& v) {
          double e = 0.0, t = 0.0;
          for (std::size_t i = 0; i < v.size(); ++i) {
            const double d = (double)v[i] - wf[i];
            e += d * d;
            t += (double)wf[i] * wf[i];
          }
          return std::sqrt(e / t);
        };
        SharedBuffer w[3], sc[3], bi[3];
        const Sc fmt[3] = {Sc::kF16, Sc::kFitF16, Sc::kFitBf16};
        bool ok = true;
        for (int f = 0; f < 3; ++f) {
          ok = ok && q.quantize_linear(w_in, src_bf16, N, K, bits, group,
                                       1.0f, w[f], sc[f], bi[f], fmt[f]);
        }
        ASSERT_TRUE(ok);
        if (!ok) { continue; }
        // The fitted codes, restated.
        std::size_t bad = 0, ties = 0;
        for (int f = 1; f < 3; ++f) {
          const bool pb = fmt[f] == Sc::kFitBf16;
          const auto* wp = static_cast<const std::uint32_t*>(w[f].contents());
          const auto* sp = static_cast<const std::uint16_t*>(sc[f].contents());
          const auto* bp = static_cast<const std::uint16_t*>(bi[f].contents());
          for (int n = 0; n < N; ++n) {
            for (int g = 0; g < G; ++g) {
              const float s1 = pb ? from_bf16(sp[n * G + g])
                                  : from_f16(sp[n * G + g]);
              const float b1 = pb ? from_bf16(bp[n * G + g])
                                  : from_f16(bp[n * G + g]);
              const float inv = s1 > 0.0f ? 1.0f / s1 : 0.0f;
              for (int i = 0; i < group; ++i) {
                const int k = g * group + i;
                const float v = wf[(std::size_t)n * K + k];
                const float t = (v - b1) * inv;
                const int want = std::clamp((int)std::round(t), 0, QMAX);
                const std::uint32_t word =
                    wp[(std::size_t)n * (K / VPW) + k / VPW];
                const int got = (int)((word >> (bits * (k % VPW))) &
                                      (std::uint32_t)QMAX);
                // At a rounding tie the GPU's reciprocal (fast math, an
                // ulp from the CPU's) may land either side.
                const float frac = t - std::floor(t);
                const bool tie = std::fabs(frac - 0.5f) < 1e-4f;
                ties += got != want && tie && std::abs(got - want) == 1;
                bad += got != want && !(tie && std::abs(got - want) == 1);
              }
            }
          }
        }
        const double e_orig_f16 = err(values(w[0], sc[0], bi[0], false,
                                             false));
        const double e_orig_bf16 = err(values(w[0], sc[0], bi[0], false,
                                              true));
        const double e_fit_f16 = err(values(w[1], sc[1], bi[1], false,
                                            false));
        const double e_fit_bf16 = err(values(w[2], sc[2], bi[2], true,
                                             true));
        std::printf("[model_quantize_roundtrip] w%d g%d from %s: bf16 run "
                    "%.3e original -> %.3e fit; f16 run %.3e -> %.3e; %zu "
                    "codes off the restatement (%zu more at a tie)\n", bits,
                    group, src_bf16 ? "bf16" : "f16", e_orig_bf16,
                    e_fit_bf16, e_orig_f16, e_fit_f16, bad, ties);
        EXPECT_TRUE(bad == 0);
        EXPECT_TRUE(e_fit_bf16 <= e_orig_bf16);
        EXPECT_TRUE(e_fit_f16 <= e_orig_f16 * 1.0001);
        // The in-memory w8 build's kernel, where model-quantize uses it.
        if (src_bf16 && bits == 8 && group == 64) {
          ComputeLibrary lib = mc->load_library("affine_dequant_bf16");
          ComputeFunction fn = lib.function("affine_quant_rows_w8g64");
          ASSERT_TRUE(fn.valid());
          SharedBuffer rw = mc->make_shared_buffer(w[2].byte_size());
          SharedBuffer rs = mc->make_shared_buffer(sc[2].byte_size());
          SharedBuffer rb = mc->make_shared_buffer(bi[2].byte_size());
          CommandStream st = mc->make_command_stream();
          {
            ComputeEncoder enc = st.begin_compute();
            const int one = 1, zero = 0;
            enc.set_function(fn);
            enc.set_buffer(0, w_in); enc.set_buffer(1, rw);
            enc.set_buffer(2, rs); enc.set_buffer(3, rb);
            enc.set_constant(4, K); enc.set_constant(5, N);
            enc.set_constant(6, one); enc.set_constant(7, zero);
            enc.dispatch({(unsigned)(K / 64), (unsigned)N, 1}, {32, 8, 1});
          }
          ASSERT_TRUE(st.commit().wait_ok());
          EXPECT_TRUE(std::memcmp(rw.contents(), w[2].contents(),
                                  rw.byte_size()) == 0 &&
                      std::memcmp(rs.contents(), sc[2].contents(),
                                  rs.byte_size()) == 0 &&
                      std::memcmp(rb.contents(), bi[2].contents(),
                                  rb.byte_size()) == 0);
        }
        // A BF16 pair through the writer and the reader, by its dtype.
        if (bits == 8 && group == 64) {
          const fs::path dir = tmp / (src_bf16 ? "bf16" : "f16");
          {
            SafetensorsWriter wr(dir.string());
            const std::int64_t wcols = (std::int64_t)K * bits / 32;
            ASSERT_TRUE(wr.add("t.weight", "U32", {N, wcols},
                               w[2].contents(), w[2].byte_size()));
            ASSERT_TRUE(wr.add("t.scales",
                               AffineQuantizer::scale_dtype(Sc::kFitBf16),
                               {N, G}, sc[2].contents(), sc[2].byte_size()));
            ASSERT_TRUE(wr.add("t.biases",
                               AffineQuantizer::scale_dtype(Sc::kFitBf16),
                               {N, G}, bi[2].contents(), bi[2].byte_size()));
            ASSERT_TRUE(wr.close());
          }
          auto wts = MetalLlamaWeights::open_model(dir.string());
          ASSERT_TRUE(wts.has_value());
          if (!wts.has_value()) { continue; }
          const auto* si = wts->info("t.scales");
          ASSERT_TRUE(si != nullptr && si->dtype == "BF16");
          SharedBuffer fs_ = wts->load("t.scales", mc);
          ASSERT_TRUE(!fs_.empty() &&
                      fs_.byte_size() == sc[2].byte_size());
          if (!fs_.empty() && fs_.byte_size() == sc[2].byte_size()) {
            EXPECT_TRUE(std::memcmp(fs_.contents(), sc[2].contents(),
                                    fs_.byte_size()) == 0);
          }
        }
      }
    }
  }
  std::error_code ec;
  fs::remove_all(tmp, ec);
}

// Real-model isolation: dequantize the quantizer's 8-bit output tensors and
// compare against the bf16 SOURCE (rel-L2). 8-bit affine should track the
// source to a few %; a GDN tensor reading ~1.0 localizes a quantizer bug in
// the gated-DeltaNet path (the in_proj_* group -- esp. the tiny a/b -- and the
// out_proj, never exercised by the dense MOSS verification). Env:
//   VPIPE_Q27_BF16  = the bf16 source dir
//   VPIPE_Q27_8BIT  = the model-quantize 8-bit output dir
// The writer must leave the DATA SECTION 16-byte aligned, because that
// one number decides whether every tensor in the shard can be handed to
// Metal as a zero-copy subview or has to be COPIED.
//
// It regressed silently before it was noticed: five shards of one
// checkpoint came out at 7, 8, 12, 11 and 15 mod 16, so `Mapped` was a
// no-op for everything this writer produced and a 22B DiT quietly
// allocated 39 GB of anonymous memory. Nothing failed; the box just
// thrashed. Hence a test on the BYTES rather than on a log line.
TEST(model_quantize_roundtrip, writer_aligns_the_data_section)
{
  namespace fs = std::filesystem;
  auto mc = std::make_unique<MetalCompute>(nullptr);
  if (!mc->valid()) { return; }               // no device: vacuous, not failed

  const fs::path tmp =
      fs::temp_directory_path() / "vpipe-st-align-test";
  fs::remove_all(tmp);
  fs::create_directories(tmp);

  // Names of DIFFERENT lengths, so the header length is not a round
  // number by luck -- an unpadded writer then lands wherever it lands.
  for (int variant = 0; variant < 4; ++variant) {
    const fs::path dir = tmp / ("v" + std::to_string(variant));
    std::vector<std::uint16_t> data(64, 0x3c00);
    {
      SafetensorsWriter wr(dir.string());
      for (int t = 0; t <= variant; ++t) {
        const std::string nm =
            "block." + std::to_string(t) + std::string((std::size_t)t + 1, 'x')
            + ".weight";
        ASSERT_TRUE(wr.add(nm, "F16", {8, 8}, data.data(),
                           data.size() * sizeof(std::uint16_t)));
      }
      ASSERT_TRUE(wr.close());
    }

    // Read the 8-byte length prefix straight off the file.
    fs::path shard;
    for (const auto& de : fs::directory_iterator(dir)) {
      if (de.path().extension() == ".safetensors") { shard = de.path(); }
    }
    ASSERT_TRUE(!shard.empty());
    std::ifstream f(shard.string(), std::ios::binary);
    ASSERT_TRUE(f.good());
    std::uint64_t hlen = 0;
    f.read(reinterpret_cast<char*>(&hlen), 8);
    const std::uint64_t data_at = 8 + hlen;
    EXPECT_TRUE((data_at % 16) == 0);

    // And the consequence that actually matters: the reader maps it
    // instead of copying. An owned buffer means it fell back.
    auto wts = MetalLlamaWeights::open_model(dir.string());
    ASSERT_TRUE(wts.has_value());
    const std::string first = "block.0x.weight";
    ASSERT_TRUE(wts->has(first));
    SharedBuffer m = wts->load_mapped(first, mc.get());
    ASSERT_TRUE(!m.empty());
    EXPECT_FALSE(m.is_owned());
  }
  fs::remove_all(tmp);
}

// An aligned data section is not enough: tensors are packed CONTIGUOUSLY,
// so one whose SIZE is not a multiple of 16 shifts every tensor after it
// off the boundary, and those are copied too.
//
// MEASURED on a quantized Gemma-4 12B before the writer held them back:
// one 2-byte `layer_scalar` per layer, 48 of them interleaved with the
// weights, plus four JSON asset blobs -- 53 tensors that between them
// misaligned 1181 of 1344. The section start was a clean 0 mod 16 the
// whole time, which is what made the old warning read as a
// contradiction.
//
// The fix is ordering, not padding, and the test pins BOTH halves of
// that: every 16-multiple-sized tensor aligned, and the byte ranges
// still tiling the blob with no gaps -- because the reference
// safetensors reader validates contiguity, so a writer that "fixed"
// alignment by leaving holes would produce files nothing else can read.
TEST(model_quantize_roundtrip, writer_defers_odd_sized_tensors)
{
  namespace fs = std::filesystem;
  auto mc = std::make_unique<MetalCompute>(nullptr);
  if (!mc->valid()) { return; }

  const fs::path dir = fs::temp_directory_path() / "vpipe-st-odd-test";
  fs::remove_all(dir);
  fs::create_directories(dir);

  std::vector<std::uint16_t> big(64, 0x3c00);      // 128 bytes, 16-multiple
  const std::uint16_t        scalar = 0x3c00;      // 2 bytes, the offender
  {
    SafetensorsWriter wr(dir.string());
    // Interleaved exactly as a real checkpoint has them.
    for (int t = 0; t < 6; ++t) {
      const std::string p = "block." + std::to_string(t) + ".";
      ASSERT_TRUE(wr.add(p + "weight", "F16", {8, 8}, big.data(),
                         big.size() * sizeof(std::uint16_t)));
      ASSERT_TRUE(wr.add(p + "layer_scalar", "F16", {1}, &scalar, 2));
    }
    ASSERT_TRUE(wr.close());
  }

  fs::path shard;
  for (const auto& de : fs::directory_iterator(dir)) {
    if (de.path().extension() == ".safetensors") { shard = de.path(); }
  }
  ASSERT_TRUE(!shard.empty());

  // Every WEIGHT maps; the 2-byte scalars are what pays for it.
  auto wts = MetalLlamaWeights::open_model(dir.string());
  ASSERT_TRUE(wts.has_value());
  for (int t = 0; t < 6; ++t) {
    const std::string nm = "block." + std::to_string(t) + ".weight";
    ASSERT_TRUE(wts->has(nm));
    SharedBuffer m = wts->load_mapped(nm, mc.get());
    ASSERT_TRUE(!m.empty());
    EXPECT_FALSE(m.is_owned());
  }

  // The bytes still tile the blob: sorted ranges start at 0, each begins
  // where the last ended. A gap here is a file the reference reader
  // rejects, which no amount of alignment makes up for.
  {
    std::ifstream f(shard.string(), std::ios::binary);
    ASSERT_TRUE(f.good());
    std::uint64_t hlen = 0;
    f.read(reinterpret_cast<char*>(&hlen), 8);
    std::string hdr((std::size_t)hlen, '\0');
    f.read(hdr.data(), (std::streamsize)hlen);
    // Ranges as [begin,end) pairs, pulled out of the header text.
    std::vector<std::pair<long long, long long>> rs;
    for (std::size_t i = hdr.find("\"data_offsets\":[");
         i != std::string::npos;
         i = hdr.find("\"data_offsets\":[", i + 1)) {
      long long b = 0, e = 0;
      if (std::sscanf(hdr.c_str() + i, "\"data_offsets\":[%lld,%lld]",
                      &b, &e) == 2) {
        rs.emplace_back(b, e);
      }
    }
    EXPECT_TRUE(rs.size() == 12);
    std::sort(rs.begin(), rs.end());
    long long last = 0;
    bool tiled = true;
    for (const auto& r : rs) {
      if (r.first != last) { tiled = false; }
      last = r.second;
    }
    EXPECT_TRUE(tiled);
  }
  fs::remove_all(dir);
}

TEST(model_quantize_roundtrip, real_gdn_dequant_vs_source)
{
  const char* srcdir = std::getenv("VPIPE_Q27_BF16");
  const char* qdir   = std::getenv("VPIPE_Q27_8BIT");
  if (srcdir == nullptr || *srcdir == '\0' || qdir == nullptr ||
      *qdir == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto ws = MetalLlamaWeights::open_model(srcdir);
  auto wq = MetalLlamaWeights::open_model(qdir);
  ASSERT_TRUE(ws.has_value() && wq.has_value());
  ComputeLibrary deq = mc->load_library("affine_dequant");

  const std::string P = "model.language_model.layers.";
  const char* tensors[] = {
    "3.self_attn.q_proj", "3.self_attn.o_proj",          // full-attn controls
    "0.linear_attn.in_proj_qkv", "0.linear_attn.in_proj_z",
    "0.linear_attn.in_proj_a", "0.linear_attn.in_proj_b",
    "0.linear_attn.out_proj"};
  for (const char* t : tensors) {
    const std::string wname = P + t + ".weight";
    const auto* si = ws->info(wname);
    if (si == nullptr || si->shape.size() != 2) {
      std::fprintf(stderr, "  [q27] %-28s: MISSING source\n", t);
      continue;
    }
    const int N = (int)si->shape[0], K = (int)si->shape[1];
    SharedBuffer sb = ws->load(wname, mc);
    std::vector<_Float16> src((std::size_t)N * K);
    const auto* p = static_cast<const std::uint16_t*>(sb.contents());
    for (std::size_t i = 0; i < src.size(); ++i) {
      std::uint32_t u = (std::uint32_t)p[i] << 16;   // bf16 -> f32 bits
      float f; std::memcpy(&f, &u, 4);
      src[i] = (_Float16)f;
    }
    SharedBuffer qw = wq->load(P + t + ".weight", mc);
    SharedBuffer qs = wq->load(P + t + ".scales", mc);
    SharedBuffer qb = wq->load(P + t + ".biases", mc);
    if (qw.empty() || qs.empty() || qb.empty()) {
      std::fprintf(stderr, "  [q27] %-28s: MISSING quant triple\n", t);
      continue;
    }
    SharedBuffer dq = dequant_(mc, deq, qw, qs, qb, N, K, 8);
    ASSERT_TRUE(!dq.empty());
    const double rl = rel_l2_(src.data(),
                             static_cast<const _Float16*>(dq.contents()),
                             (std::size_t)N * K);
    std::fprintf(stderr, "  [q27] %-28s N=%d K=%d rel-L2=%.4f\n", t, N, K, rl);
    EXPECT_TRUE(rl < 0.2);   // 8-bit affine tracks the source within a few %
  }
}

// MoE bridge isolation: dequantize the quantizer's switch_mlp.* 3D expert
// tensors (+ the w8 router and the shared expert) and compare against the
// raw-HF bf16 SOURCE -- proving experts.gate_up_proj was split (gate=rows
// [0:I], up=rows [I:2I]) and quantized per expert into the loader's
// switch_mlp.gate_proj / up_proj / down_proj layout. NO model load. Env:
//   VPIPE_QMOE_BF16 = the raw-HF MoE source dir
//   VPIPE_QMOE_8BIT = the model-quantize 8-bit output dir
TEST(model_quantize_roundtrip, real_moe_dequant_vs_source)
{
  const char* srcdir = std::getenv("VPIPE_QMOE_BF16");
  const char* qdir   = std::getenv("VPIPE_QMOE_8BIT");
  if (srcdir == nullptr || *srcdir == '\0' || qdir == nullptr ||
      *qdir == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto ws = MetalLlamaWeights::open_model(srcdir);
  auto wq = MetalLlamaWeights::open_model(qdir);
  ASSERT_TRUE(ws.has_value() && wq.has_value());
  ComputeLibrary deq = mc->load_library("affine_dequant");

  // bf16 source tensor -> f16 host vector.
  auto src_f16 = [&](const std::string& nm, std::vector<_Float16>& out)
      -> const MetalLlamaWeights::TensorInfo* {
    const auto* si = ws->info(nm);
    if (si == nullptr) { return nullptr; }
    SharedBuffer sb = ws->load(nm, mc);
    if (sb.empty()) { return nullptr; }
    std::size_t n = 1;
    for (auto d : si->shape) { n *= (std::size_t)d; }
    out.resize(n);
    const auto* p = static_cast<const std::uint16_t*>(sb.contents());
    for (std::size_t i = 0; i < n; ++i) {
      std::uint32_t u = (std::uint32_t)p[i] << 16;
      float f; std::memcpy(&f, &u, 4);
      out[i] = (_Float16)f;
    }
    return si;
  };

  const std::string P = "model.language_model.layers.0.";
  const int experts[] = {0, 7, 255};

  // ---- experts.gate_up_proj [E,2I,H] -> switch_mlp.{gate,up}_proj [E,I,H] --
  {
    std::vector<_Float16> gu;
    const auto* gi = src_f16(P + "mlp.experts.gate_up_proj", gu);
    ASSERT_TRUE(gi != nullptr && gi->shape.size() == 3);
    const int E = (int)gi->shape[0], twoI = (int)gi->shape[1];
    const int H = (int)gi->shape[2], I = twoI / 2;
    for (int which = 0; which < 2; ++which) {     // 0=gate, 1=up
      const char* leaf = which == 0 ? "mlp.switch_mlp.gate_proj"
                                    : "mlp.switch_mlp.up_proj";
      SharedBuffer qw = wq->load(P + leaf + ".weight", mc);
      SharedBuffer qs = wq->load(P + leaf + ".scales", mc);
      SharedBuffer qb = wq->load(P + leaf + ".biases", mc);
      ASSERT_TRUE(!qw.empty() && !qs.empty() && !qb.empty());
      SharedBuffer dq = dequant_(mc, deq, qw, qs, qb, E * I, H, 8);
      ASSERT_TRUE(!dq.empty());
      const auto* dp = static_cast<const _Float16*>(dq.contents());
      for (int e : experts) {
        const _Float16* src_e =
            gu.data() + (std::size_t)e * twoI * H +
            (std::size_t)(which == 0 ? 0 : I) * H;
        const _Float16* dq_e = dp + (std::size_t)e * I * H;
        const double rl = rel_l2_(src_e, dq_e, (std::size_t)I * H);
        std::fprintf(stderr, "  [qmoe] %-22s e=%-3d rel-L2=%.4f\n",
                     leaf, e, rl);
        EXPECT_TRUE(rl < 0.03);
      }
    }
  }

  // ---- experts.down_proj [E,H,I] -> switch_mlp.down_proj [E,H,I] --------
  {
    std::vector<_Float16> dn;
    const auto* di = src_f16(P + "mlp.experts.down_proj", dn);
    ASSERT_TRUE(di != nullptr && di->shape.size() == 3);
    const int E = (int)di->shape[0], H = (int)di->shape[1];
    const int I = (int)di->shape[2];
    SharedBuffer qw = wq->load(P + "mlp.switch_mlp.down_proj.weight", mc);
    SharedBuffer qs = wq->load(P + "mlp.switch_mlp.down_proj.scales", mc);
    SharedBuffer qb = wq->load(P + "mlp.switch_mlp.down_proj.biases", mc);
    ASSERT_TRUE(!qw.empty() && !qs.empty() && !qb.empty());
    SharedBuffer dq = dequant_(mc, deq, qw, qs, qb, E * H, I, 8);
    ASSERT_TRUE(!dq.empty());
    const auto* dp = static_cast<const _Float16*>(dq.contents());
    for (int e : experts) {
      const _Float16* src_e = dn.data() + (std::size_t)e * H * I;
      const _Float16* dq_e = dp + (std::size_t)e * H * I;
      const double rl = rel_l2_(src_e, dq_e, (std::size_t)H * I);
      std::fprintf(stderr, "  [qmoe] %-22s e=%-3d rel-L2=%.4f\n",
                   "switch_mlp.down_proj", e, rl);
      EXPECT_TRUE(rl < 0.03);
    }
  }

  // ---- router (mlp.gate, w8) + shared_expert.gate_proj (2D) ------------
  struct Two { const char* nm; } twos[] = {
    {"mlp.gate"}, {"mlp.shared_expert.gate_proj"},
    {"mlp.shared_expert_gate"}};
  for (const auto& t : twos) {
    std::vector<_Float16> src;
    const auto* si = src_f16(P + t.nm + ".weight", src);
    ASSERT_TRUE(si != nullptr && si->shape.size() == 2);
    const int N = (int)si->shape[0], K = (int)si->shape[1];
    SharedBuffer qw = wq->load(P + t.nm + ".weight", mc);
    SharedBuffer qs = wq->load(P + t.nm + ".scales", mc);
    SharedBuffer qb = wq->load(P + t.nm + ".biases", mc);
    ASSERT_TRUE(!qw.empty() && !qs.empty() && !qb.empty());
    // bits = packed-u32 cols * 32 / K (w8 for router/shared-gate, opt.bits
    // for the shared-expert projection).
    const auto* wi = wq->info(P + t.nm + ".weight");
    const int bits = (int)((std::int64_t)wi->shape[1] * 32 / K);
    SharedBuffer dq = dequant_(mc, deq, qw, qs, qb, N, K, bits);
    ASSERT_TRUE(!dq.empty());
    const double rl = rel_l2_(src.data(),
                              static_cast<const _Float16*>(dq.contents()),
                              (std::size_t)N * K);
    std::fprintf(stderr, "  [qmoe] %-22s N=%d K=%d w%d rel-L2=%.4f\n",
                 t.nm, N, K, bits, rl);
    EXPECT_TRUE(rl < (bits == 8 ? 0.03 : 0.2));
  }
}
