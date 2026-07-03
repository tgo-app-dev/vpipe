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

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// Real-model isolation: dequantize the quantizer's 8-bit output tensors and
// compare against the bf16 SOURCE (rel-L2). 8-bit affine should track the
// source to a few %; a GDN tensor reading ~1.0 localizes a quantizer bug in
// the gated-DeltaNet path (the in_proj_* group -- esp. the tiny a/b -- and the
// out_proj, never exercised by the dense MOSS verification). Env:
//   VPIPE_Q27_BF16  = the bf16 source dir
//   VPIPE_Q27_8BIT  = the model-quantize 8-bit output dir
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
