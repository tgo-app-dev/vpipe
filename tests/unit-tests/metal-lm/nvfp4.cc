// nvfp4.cc -- ModelOpt NVFP4 weights (generative-models/shared/nvfp4.h):
// the format decoded as ModelOpt's NVFP4QTensor.dequantize does it, the
// served super-blocks, the kernels against the CPU reference, and -- on a
// real checkpoint -- its projections read as its base model's, the native
// k-quant route token-exact against the dense dequantized one, and the
// plan's booking against the load.
//
// Env: VPIPE_NVFP4_TEST_MODEL_PATH (an NVFP4 Qwen3.5-family checkpoint:
// MiniMaxAI/MiniMax-H3-Image-PE, a Qwen3.8-27B fine-tune) and
// VPIPE_NVFP4_BASE_MODEL_PATH (its base in MLX affine: Qwen3.8-27B-4bit
// or -OptiQ-4bit), for the base test.

#include "tests/unit-tests/metal-lm/metal-lm-test-common.h"

#include "common/media-line.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/lm-run-memory.h"
#include "generative-models/shared/fp8.h"
#include "generative-models/shared/nvfp4.h"
#include "stages/model-memory.h"

#include <filesystem>
#include <tuple>

namespace {

namespace nvfp4 = genai::nvfp4;

// A deterministic stream of bytes.
struct Lcg {
  std::uint32_t s;
  std::uint32_t next() { s = s * 1664525u + 1013904223u; return s >> 8; }
};

// An E4M3 block scale in the range a real checkpoint's take (exponents
// 1..14, so neither subnormal nor NaN), with a subnormal now and then.
std::uint8_t
scale_byte_(Lcg& r)
{
  const std::uint32_t v = r.next();
  if (v % 61 == 0) { return (std::uint8_t)(1 + v % 7); }   // subnormal
  return (std::uint8_t)((1 + v % 14) << 3 | (v >> 8) % 8);
}

struct Weight {
  int N = 0, K = 0;
  std::vector<std::uint8_t> codes, scales;
  float g = 0.0f;
};

Weight
random_weight_(int N, int K, std::uint32_t seed)
{
  Weight w;
  w.N = N;
  w.K = K;
  w.g = 1.7e-4f;
  Lcg r{seed};
  w.codes.resize((std::size_t)N * K / 2);
  for (auto& b : w.codes) { b = (std::uint8_t)r.next(); }
  w.scales.resize((std::size_t)N * K / nvfp4::kBlock);
  for (auto& b : w.scales) { b = scale_byte_(r); }
  return w;
}

double
cosine_(const std::vector<float>& a, const std::vector<float>& b)
{
  double ab = 0, aa = 0, bb = 0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    ab += (double)a[i] * b[i];
    aa += (double)a[i] * a[i];
    bb += (double)b[i] * b[i];
  }
  return aa > 0 && bb > 0 ? ab / std::sqrt(aa * bb) : 0.0;
}

}  // namespace

// The two tables and the packing, as ModelOpt's NVFP4QTensor: E2M1 codes
// {0, .5, 1, 1.5, 2, 3, 4, 6} with bit 3 the sign; float8_e4m3fn scales;
// input 2i the LOW nibble of byte i, 2i+1 the high; w = e2m1 * (scale *
// scale_2). The served super-block reads every input as the file does.
TEST(nvfp4, decodes_as_modelopt_dequantizes)
{
  EXPECT_TRUE(nvfp4::e4m3_to_float(0x38) == 1.0f);
  EXPECT_TRUE(nvfp4::e4m3_to_float(0x7E) == 448.0f);
  EXPECT_TRUE(nvfp4::e4m3_to_float(0x08) == std::ldexp(1.0f, -6));
  EXPECT_TRUE(nvfp4::e4m3_to_float(0x01) == std::ldexp(1.0f, -9));
  EXPECT_TRUE(nvfp4::e4m3_to_float(0x07) == 7.0f * std::ldexp(1.0f, -9));
  EXPECT_TRUE(nvfp4::e4m3_to_float(0xB8) == -1.0f);
  EXPECT_TRUE(nvfp4::e4m3_to_float(0x3C) == 1.5f);
  EXPECT_TRUE(std::isnan(nvfp4::e4m3_to_float(0x7F)));
  const float want[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6,
                          -0.0f, -.5f, -1, -1.5f, -2, -3, -4, -6};
  for (int c = 0; c < 16; ++c) {
    EXPECT_TRUE(nvfp4::e2m1_value((std::uint8_t)c) == want[c]);
  }

  // One row of 512: byte i = (hi << 4) | lo, so input 2i reads lo.
  Weight w = random_weight_(1, 512, 7);
  std::vector<float> ref(512);
  nvfp4::dequantize(w.codes.data(), w.scales.data(), w.g, 1, 512,
                    ref.data());
  for (int k = 0; k < 512; ++k) {
    const std::uint8_t b = w.codes[(std::size_t)k / 2];
    const int code = (k & 1) != 0 ? b >> 4 : b & 0xF;
    const float s = nvfp4::e4m3_to_float(w.scales[(std::size_t)k / 16]);
    EXPECT_TRUE(ref[(std::size_t)k] == want[code] * (s * w.g));
  }
  std::vector<std::uint8_t> served(nvfp4::served_bytes(1, 512));
  ASSERT_TRUE(nvfp4::pack(w.codes.data(), w.scales.data(), w.g, 1, 512,
                          served.data()));
  int same = 0;
  for (int k = 0; k < 512; ++k) {
    const float v = nvfp4::served_value(
        served.data() + (std::size_t)(k / 256) * nvfp4::kSuperBytes, k & 255);
    same += v == ref[(std::size_t)k];
  }
  EXPECT_TRUE(same == 512);
  // K must fill whole super-blocks.
  EXPECT_TRUE(!nvfp4::pack(w.codes.data(), w.scales.data(), w.g, 1, 496,
                           served.data()));
}

// The kernels against the CPU reference at the shapes the 27B runs (K 5120
// and 17408), an odd row count for the row guard: dequant_nvfp4_f16 is the
// f32 value rounded once (the same products in the same order: exact),
// qmv_nvfp4_f16 the f64 dot within f16 output rounding, and the batched
// pair each row bit for bit the single-row kernel.
TEST(metal_lm_smoke, nvfp4_kernels_match_cpu)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  for (const char* f : {"dequant_nvfp4_f16", "qmv_nvfp4_f16",
                        "qmv_nvfp4_batch_f16", "qmv_nvfp4_batch4_f16"}) {
    ASSERT_TRUE(lib.function(f).valid());
  }
  for (const int K : {5120, 17408}) {
    const int N = 97;
    Weight w = random_weight_(N, K, (std::uint32_t)K);
    std::vector<std::uint8_t> served(nvfp4::served_bytes(N, K));
    ASSERT_TRUE(nvfp4::pack(w.codes.data(), w.scales.data(), w.g, N, K,
                            served.data()));
    std::vector<float> deq((std::size_t)N * K);
    nvfp4::dequantize(w.codes.data(), w.scales.data(), w.g, N, K,
                      deq.data());
    auto wbuf = mc->make_shared_buffer(served.size());
    std::memcpy(wbuf.contents(), served.data(), served.size());

    // dequant
    auto dbuf = mc->make_shared_buffer((std::size_t)N * K * 2);
    {
      auto st = mc->make_command_stream();
      { auto enc = st.begin_compute();
        enc.set_function(lib.function("dequant_nvfp4_f16"));
        enc.set_buffer(0, wbuf); enc.set_buffer(1, dbuf);
        enc.set_constant(2, N * K);
        enc.dispatch({(unsigned)(N * K), 1, 1}, {256, 1, 1}); }
      st.commit().wait();
    }
    const auto* dp = static_cast<const _Float16*>(dbuf.contents());
    std::size_t exact = 0;
    for (std::size_t i = 0; i < deq.size(); ++i) {
      exact += dp[i] == (_Float16)deq[i];
    }
    std::printf("[nvfp4] K %5d dequant: %zu of %zu exact\n", K, exact,
                deq.size());
    EXPECT_TRUE(exact == deq.size());

    // qmv, one row of x
    constexpr int M = 3;
    std::vector<_Float16> x((std::size_t)M * K);
    Lcg r{99u + (std::uint32_t)K};
    for (auto& v : x) {
      v = (_Float16)(((float)(r.next() % 2001) - 1000.0f) / 400.0f);
    }
    auto xbuf = mc->make_shared_buffer(x.size() * 2);
    std::memcpy(xbuf.contents(), x.data(), x.size() * 2);
    auto y1 = mc->make_shared_buffer((std::size_t)M * N * 2);
    auto y2 = mc->make_shared_buffer((std::size_t)M * N * 2);
    auto y4 = mc->make_shared_buffer((std::size_t)M * N * 2);
    {
      auto st = mc->make_command_stream();
      { auto enc = st.begin_compute();
        for (int m = 0; m < M; ++m) {
          enc.set_function(lib.function("qmv_nvfp4_f16"));
          enc.set_buffer(0, wbuf);
          enc.set_buffer(1, xbuf, (std::size_t)m * K * 2);
          enc.set_buffer(2, y1, (std::size_t)m * N * 2);
          enc.set_constant(3, K); enc.set_constant(4, N);
          enc.dispatch({32, (unsigned)(((N + 1) / 2) * 2), 1}, {32, 2, 1});
        }
        for (const auto& [fn, y, maxm] :
             {std::tuple{"qmv_nvfp4_batch_f16", &y2, 2},
              std::tuple{"qmv_nvfp4_batch4_f16", &y4, 4}}) {
          enc.set_function(lib.function(fn));
          enc.set_buffer(0, wbuf); enc.set_buffer(1, xbuf);
          enc.set_buffer(2, *y);
          enc.set_constant(3, K); enc.set_constant(4, N);
          enc.set_constant(5, M); enc.set_constant(6, N);
          enc.set_constant(7, 0);
          enc.dispatch({32, (unsigned)(((N + 1) / 2) * 2),
                        (unsigned)((M + maxm - 1) / maxm)}, {32, 2, 1});
        } }
      st.commit().wait();
    }
    const auto* p1 = static_cast<const _Float16*>(y1.contents());
    double maxrel = 0.0;
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
          acc += (double)(float)x[(std::size_t)m * K + k] *
                 (double)deq[(std::size_t)n * K + k];
        }
        const double d = std::fabs((double)(float)p1[m * N + n] - acc);
        maxrel = std::fmax(maxrel, d / (std::fabs(acc) + 1e-2));
      }
    }
    const bool b2 = std::memcmp(y1.contents(), y2.contents(),
                                (std::size_t)M * N * 2) == 0;
    const bool b4 = std::memcmp(y1.contents(), y4.contents(),
                                (std::size_t)M * N * 2) == 0;
    std::printf("[nvfp4] K %5d qmv: max rel err %.3g; batch2 %s, batch4 %s "
                "the single-row kernel\n", K, maxrel,
                b2 ? "==" : "!=", b4 ? "==" : "!=");
    EXPECT_TRUE(maxrel < 2e-3);
    EXPECT_TRUE(b2 && b4);
  }
}

// nvfp4_serve -- the GPU build a STREAMED NVFP4 layer's products come
// from -- writes the bytes the host pack a resident load takes writes:
// nvfp4::pack over ModelOpt's layout, ComfyUI's spelling normalized (the
// even column's nibble high, the block scales swizzled into 128 x 4 tiles,
// padded) and, with a row multiplier, scale_2 * mul[n] in f32 as each
// row's tensor scale (down_proj's AWQ scale folded into up_proj). 200 rows
// cross a tile and pad the last; written at an offset, as k into q|k.
TEST(metal_lm_smoke, nvfp4_serve_is_the_host_pack)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise_bf16");
  auto fn = lib.function("nvfp4_serve");
  ASSERT_TRUE(fn.valid());
  if (!fn.valid()) { return; }
  int checked = 0;
  for (const int K : {512, 768}) {
    for (const bool comfy : {false, true}) {
      for (const bool awq : {false, true}) {
        const int N = 200, B = K / nvfp4::kBlock;
        Weight w = random_weight_(N, K, (std::uint32_t)(K + comfy + 2 * awq));
        std::vector<float> mul;
        Lcg r{7u};
        for (int n = 0; n < N && awq; ++n) {
          mul.push_back(0.25f + (float)(r.next() % 1000) / 400.0f);
        }
        // The host's served form, as LayerLoad::nvfp4_served builds it.
        std::vector<std::uint8_t> want(nvfp4::served_bytes(N, K));
        ASSERT_TRUE(nvfp4::pack(w.codes.data(), w.scales.data(), w.g, N, K,
                                want.data()));
        const int supers = K / nvfp4::kSuper;
        for (int n = 0; n < N && awq; ++n) {
          const float gg = w.g * mul[(std::size_t)n];
          for (int sb = 0; sb < supers; ++sb) {
            std::memcpy(want.data() + ((std::size_t)n * supers + sb) *
                                          nvfp4::kSuperBytes,
                        &gg, 4);
          }
        }
        // The file's tensors in its spelling.
        std::vector<std::uint8_t> codes = w.codes;
        std::vector<std::uint8_t> blocks = w.scales;
        if (comfy) {
          for (auto& b : codes) { b = (std::uint8_t)(b << 4 | b >> 4); }
          const std::size_t Np = (std::size_t)(N + 127) / 128 * 128;
          const std::size_t Bp = (std::size_t)(B + 3) / 4 * 4;
          blocks.assign(Np * Bp, 0);
          for (int n = 0; n < N; ++n) {
            for (int b = 0; b < B; ++b) {
              blocks[genai::fp8::nvfp4_scale_index(
                  (std::size_t)n, (std::size_t)b, (std::size_t)B, true)] =
                  w.scales[(std::size_t)n * B + b];
            }
          }
        }
        auto cb = mc->make_shared_buffer(codes.size());
        auto bb = mc->make_shared_buffer(blocks.size());
        auto gb = mc->make_shared_buffer(4);
        auto mb = mc->make_shared_buffer(4 * (std::size_t)N);
        std::memcpy(cb.contents(), codes.data(), codes.size());
        std::memcpy(bb.contents(), blocks.data(), blocks.size());
        std::memcpy(gb.contents(), &w.g, 4);
        if (awq) { std::memcpy(mb.contents(), mul.data(), 4 * (std::size_t)N); }
        const std::size_t off = nvfp4::served_bytes(3, K);
        auto out = mc->make_shared_buffer(off + want.size());
        std::memset(out.contents(), 0xA5, out.byte_size());
        const std::uint32_t flags =
            (comfy ? 3u : 0u) | (awq ? 4u : 0u);
        {
          auto st = mc->make_command_stream();
          { auto enc = st.begin_compute();
            enc.set_function(fn);
            enc.set_buffer(0, cb); enc.set_buffer(1, bb);
            enc.set_buffer(2, gb); enc.set_buffer(3, mb);
            enc.set_buffer(4, out, off);
            enc.set_constant(5, N); enc.set_constant(6, K);
            enc.set_constant(7, flags);
            enc.dispatch({(unsigned)supers, (unsigned)N, 1}, {8, 32, 1}); }
          st.commit().wait();
        }
        const auto* o = static_cast<const std::uint8_t*>(out.contents());
        bool head_kept = true;
        for (std::size_t i = 0; i < off; ++i) { head_kept &= o[i] == 0xA5; }
        const bool same = std::memcmp(o + off, want.data(), want.size()) == 0;
        std::printf("[nvfp4] serve K %d %s%s: %s\n", K,
                    comfy ? "ComfyUI" : "ModelOpt", awq ? " + AWQ rows" : "",
                    same && head_kept ? "the host pack's bytes"
                                      : "DIFFERENT");
        EXPECT_TRUE(same && head_kept);
        ++checked;
      }
    }
  }
  EXPECT_TRUE(checked == 8);
}

// A real NVFP4 checkpoint read as its own base: a fine-tune's projection
// lies close to the base model's (cosine near 1), so a decode that gets
// the format right matches it and one that swaps the nibbles -- or the
// scale's place -- does not. Base in MLX affine (w = scale q + bias).
// MEASURED on MiniMax-H3-Image-PE against Qwen3.8-27B-OptiQ-4bit (its
// projections 8-bit there): 0.9959-0.9964, nibbles swapped -0.002-0.002,
// scales one block on 0.910-0.922.
TEST(nvfp4, a_real_projection_reads_as_its_base)
{
  const char* pe = std::getenv("VPIPE_NVFP4_TEST_MODEL_PATH");
  const char* bp = std::getenv("VPIPE_NVFP4_BASE_MODEL_PATH");
  if (pe == nullptr || *pe == 0 || bp == nullptr || *bp == 0) { return; }
  auto wq = genai::MetalLlamaWeights::open_model(pe);
  auto wb = genai::MetalLlamaWeights::open_model(bp);
  ASSERT_TRUE(wq.has_value() && wb.has_value());
  if (!wq || !wb) { return; }
  for (const char* part : {"layers.0.mlp.gate_proj", "layers.0.mlp.down_proj",
                           "layers.3.self_attn.q_proj",
                           "layers.0.linear_attn.in_proj_qkv",
                           "layers.63.mlp.up_proj"}) {
    const std::string qn = std::string("model.language_model.") + part;
    const std::string bn = std::string("language_model.model.") + part;
    const auto* wi = wq->info(qn + ".weight");
    const auto* si = wq->info(qn + ".weight_scale");
    const auto* bw = wb->info(bn + ".weight");
    const auto* bs = wb->info(bn + ".scales");
    ASSERT_TRUE(wi && si && bw && bs);
    if (!wi || !si || !bw || !bs) { return; }
    const int K = (int)wi->shape[1] * 2;
    const int rows = std::min<int>(256, (int)wi->shape[0]);
    // The first `rows` rows of each.
    std::vector<std::uint8_t> codes((std::size_t)wi->shape[0] * K / 2),
        scales((std::size_t)si->shape[0] * si->shape[1]);
    float g = 0.0f;
    ASSERT_TRUE(wq->pread_into(qn + ".weight", codes.data(), codes.size(),
                               true) &&
                wq->pread_into(qn + ".weight_scale", scales.data(),
                               scales.size(), true) &&
                wq->pread_into(qn + ".weight_scale_2", &g, 4, true));
    std::vector<float> ours((std::size_t)rows * K);
    nvfp4::dequantize(codes.data(), scales.data(), g, rows, K, ours.data());
    // MUTATIONS: the nibbles swapped; every block's scale one block on.
    const std::size_t nb = (std::size_t)rows * K / 2;
    std::vector<std::uint8_t> swapped(codes.begin(), codes.begin() + nb);
    for (auto& b : swapped) { b = (std::uint8_t)((b << 4) | (b >> 4)); }
    std::vector<float> sw((std::size_t)rows * K);
    nvfp4::dequantize(swapped.data(), scales.data(), g, rows, K, sw.data());
    std::vector<std::uint8_t> shifted(scales.begin(),
                                      scales.begin() +
                                          (std::size_t)rows * K / 16);
    std::rotate(shifted.begin(), shifted.begin() + 1, shifted.end());
    std::vector<float> sh((std::size_t)rows * K);
    nvfp4::dequantize(codes.data(), shifted.data(), g, rows, K, sh.data());

    const int bits = (int)(bw->shape[1] * 32 / K);
    const int group = K / (int)bs->shape[1];
    std::vector<std::uint32_t> pk((std::size_t)bw->shape[0] * bw->shape[1]);
    std::vector<std::uint16_t> bsc((std::size_t)bs->shape[0] * bs->shape[1]),
        bbi(bsc.size());
    ASSERT_TRUE(wb->pread_into(bn + ".weight", pk.data(), pk.size() * 4,
                               true) &&
                wb->pread_into(bn + ".scales", bsc.data(), bsc.size() * 2,
                               true) &&
                wb->pread_into(bn + ".biases", bbi.data(), bbi.size() * 2,
                               true));
    const bool sbf = bs->dtype == "BF16";
    auto f16 = [&](std::uint16_t v) {
      if (sbf) {
        std::uint32_t u = (std::uint32_t)v << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
      }
      _Float16 h;
      std::memcpy(&h, &v, 2);
      return (float)h;
    };
    std::vector<float> base((std::size_t)rows * K);
    const int per = 32 / bits;
    const std::uint32_t mask = (1u << bits) - 1u;
    for (int n = 0; n < rows; ++n) {
      for (int k = 0; k < K; ++k) {
        const std::uint32_t word =
            pk[(std::size_t)n * bw->shape[1] + k / per];
        const float q = (float)((word >> (bits * (k % per))) & mask);
        const std::size_t gi = (std::size_t)n * bs->shape[1] + k / group;
        base[(std::size_t)n * K + k] = f16(bsc[gi]) * q + f16(bbi[gi]);
      }
    }
    const double c = cosine_(ours, base), cs = cosine_(sw, base),
                 ch = cosine_(sh, base);
    std::printf("[nvfp4] %s [%d of %lld x %d], base %d-bit: cosine %.4f "
                "(nibbles swapped %.4f, scales shifted %.4f)\n", part, rows,
                (long long)wi->shape[0], K, bits, c, cs, ch);
    // A neighbour's block scale is close to a block's own (0.91-0.92 on
    // the expander's projections), so that mutation only has to land
    // clearly short of the right reading.
    EXPECT_TRUE(c > 0.99 && cs < 0.1 && ch < c - 0.03);
  }
}

namespace {

struct Run {
  std::vector<std::int32_t> ids;
  std::vector<float> first;      // the prompt's next-token logits
  double load_s = 0.0, tok_s = 0.0;
};

// Load `path` (dense when `dense`), prefill a prompt, decode `steps`
// greedily.
std::optional<Run>
greedy_(const std::string& path, bool dense, int steps)
{
  if (dense) {
    setenv("VPIPE_NVFP4_DENSE", "1", 1);
  } else {
    unsetenv("VPIPE_NVFP4_DENSE");
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return std::nullopt; }
  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  if (!cfg) { return std::nullopt; }
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 256;
  mcfg.max_pages = 8;
  const auto t0 = std::chrono::steady_clock::now();
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  unsetenv("VPIPE_NVFP4_DENSE");
  if (!model) { return std::nullopt; }
  Run r;
  r.load_s = std::chrono::duration<double>(
                 std::chrono::steady_clock::now() - t0).count();
  std::printf("[nvfp4] %s: %s, loaded in %.1f s\n",
              dense ? "dense" : "native",
              model->uses_kquant() ? "k-quant route" : "dense route",
              r.load_s);
  auto tok = genai::Tokenizer::from_huggingface_json(path + "/tokenizer.json",
                                                     &sess);
  if (!tok) { return std::nullopt; }
  auto* cm = model->context_manager();
  auto root = cm->acquire_root();
  r.first = model->prefill(root, tok->encode(
      "A short, vivid description of a lighthouse on a stormy night:"));
  if (r.first.empty()) { return std::nullopt; }
  std::int32_t next = 0;
  for (std::size_t v = 1; v < r.first.size(); ++v) {
    if (r.first[v] > r.first[(std::size_t)next]) { next = (std::int32_t)v; }
  }
  r.ids.push_back(next);
  const auto t1 = std::chrono::steady_clock::now();
  for (int i = 1; i < steps && next >= 0; ++i) {
    next = model->decode_step_fast(root, next);
    r.ids.push_back(next);
  }
  r.tok_s = (double)(r.ids.size() - 1) /
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t1).count();
  std::printf("[nvfp4] %s greedy (%.1f tok/s): '%s'\n",
              dense ? "dense" : "native", r.tok_s,
              tok->decode(r.ids).c_str());
  cm->release(root);
  return r;
}

}  // namespace

// The native k-quant route against the dense one -- every NVFP4 weight
// dequantized to f16 at load, the verified dense Qwen3.5 path: greedy
// token-exact over 48 tokens, the prompt's logits within f16's rounding.
TEST(nvfp4, native_decodes_as_the_dense_reference)
{
  const char* pe = std::getenv("VPIPE_NVFP4_TEST_MODEL_PATH");
  if (pe == nullptr || *pe == 0) { return; }
  constexpr int kSteps = 48;
  auto nat = greedy_(pe, false, kSteps);
  ASSERT_TRUE(nat.has_value());
  auto den = greedy_(pe, true, kSteps);
  ASSERT_TRUE(den.has_value());
  if (!nat || !den) { return; }
  double num = 0, d2 = 0;
  for (std::size_t v = 0; v < nat->first.size(); ++v) {
    const double e = (double)nat->first[v] - den->first[v];
    num += e * e;
    d2 += (double)den->first[v] * den->first[v];
  }
  const double rel = std::sqrt(num / std::max(d2, 1e-30));
  int same = 0;
  while (same < (int)nat->ids.size() && same < (int)den->ids.size() &&
         nat->ids[(std::size_t)same] == den->ids[(std::size_t)same]) {
    ++same;
  }
  std::printf("[nvfp4] native vs dense: logits rel-L2 %.5f, %d of %d tokens "
              "the same before the first difference; decode %.1f vs %.1f "
              "tok/s\n", rel, same, kSteps, nat->tok_s, den->tok_s);
  EXPECT_TRUE(rel < 0.01);
  EXPECT_TRUE(same == kSteps);
}

// The plan books what the native load holds: dir_weights_bytes (the files)
// against the device's allocation over the model's load and its tower's.
// The served form is 4.625 bits a weight against the file's 4.5.
TEST(nvfp4, the_plan_books_what_the_load_holds)
{
  const char* path = std::getenv("VPIPE_NVFP4_TEST_MODEL_PATH");
  if (path == nullptr || *path == 0) { return; }
  // The weights as the plan books them (dir_weights_bytes: the NVFP4
  // linears at their served size, 148 bytes a 256 inputs against the
  // file's 144) and, apart, what the load runs in (the lm:<dir> arena, at
  // this test's pool).
  const std::size_t booked = model_memory::dir_weights_bytes(path);
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  genai::ModelLoader loader(&sess);
  auto w = loader.load(path);
  ASSERT_TRUE(w.has_value());
  if (!w) { return; }
  std::size_t tower = 0, file = 0;
  if (auto wts = genai::MetalLlamaWeights::open_model(path)) {
    for (const std::string& nm : wts->tensor_names()) {
      const auto* ti = wts->info(nm);
      if (ti == nullptr) { continue; }
      file += (std::size_t)ti->nbytes;
      if (nm.find(".visual.") != std::string::npos) {
        tower += (std::size_t)ti->nbytes;
      }
    }
  }
  genai::LmRunShape shape;
  shape.page_tokens = 256;
  shape.max_pages = 2;
  shape.kv_tokens = 512;
  const std::size_t run = genai::lm_run_bytes(&sess, path, shape);
  auto mcfg = genai::MetalQwenModel::config_from(w->config);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = shape.page_tokens;
  mcfg.max_pages = (int)shape.max_pages;
  const std::size_t a0 = mc->memory_budget_settled().allocated;
  auto m = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(m != nullptr);
  if (!m) { return; }
  EXPECT_TRUE(m->uses_kquant());
  const std::size_t a1 = mc->memory_budget_settled().allocated;
  std::unique_ptr<genai::MetalQwenVisionEncoder> vis;
  if (w->config.vision.present) {
    auto vc = genai::MetalQwenVisionEncoder::config_from(w->config);
    vis = genai::MetalQwenVisionEncoder::load(path, mc, vc);
  }
  const std::size_t a2 = mc->memory_budget_settled().allocated;
  const std::size_t model = a1 - a0, held_tower = a2 - a1;
  const std::size_t booked_model = booked - tower;
  std::printf("[nvfp4] booked %.0f MB (file %.0f, tower %.0f), run arena "
              "%.0f MB; held: model %.0f MB (%.0f past its booked weights), "
              "tower %.0f MB\n", booked / 1e6, file / 1e6, tower / 1e6,
              run / 1e6, model / 1e6,
              ((double)model - (double)booked_model) / 1e6,
              held_tower / 1e6);
  EXPECT_TRUE(vis != nullptr || !w->config.vision.present);
  // The weights at least what was booked for them, and what is past them
  // inside the run arena booked beside them; the tower as stored.
  EXPECT_TRUE(model >= booked_model && model <= booked_model + run);
  EXPECT_TRUE(held_tower <= tower + ((std::size_t)16 << 20));
}

namespace {

// Python's str.format with one positional argument: "{{" / "}}" are
// braces, "{}" the argument -- how the expander's script fills its task
// prompt.
std::string
py_format_(const std::string& tpl, const std::string& arg)
{
  std::string out;
  for (std::size_t i = 0; i < tpl.size(); ++i) {
    if (tpl.compare(i, 2, "{{") == 0) { out += '{'; ++i; continue; }
    if (tpl.compare(i, 2, "}}") == 0) { out += '}'; ++i; continue; }
    if (tpl.compare(i, 2, "{}") == 0) { out += arg; ++i; continue; }
    out += tpl[i];
  }
  return out;
}

std::string
read_text_(const std::filesystem::path& p)
{
  std::ifstream f(p);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace

// MiniMax-H3-Image-PE end to end, as its release's script asks it
// (scripts/expand_prompt.py): its system prompt, its task prompt with the
// release's own text-to-image instruction, thinking on -- through the
// model manager, as text-chat loads it. After its reasoning it must
// answer the one JSON object it was told to, a caption and a ratio.
TEST(nvfp4, the_expander_writes_its_caption)
{
  const char* path = std::getenv("VPIPE_NVFP4_TEST_MODEL_PATH");
  if (path == nullptr || *path == 0) { return; }
  const std::filesystem::path dir(path);
  const std::string sys = read_text_(dir / "prompt/t2i/system_prompt.txt");
  const std::string task = read_text_(dir / "prompt/t2i/task_prompt.txt");
  if (sys.empty() || task.empty()) { return; }   // not the expander
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (mgr == nullptr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  spec.max_pages = 32;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  if (!lm || !lm->valid()) { return; }
  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);
  std::vector<std::int32_t> ids;
  std::string s = sys;
  while (!s.empty() && s.back() == '\n') { s.pop_back(); }
  ASSERT_TRUE(tpl->render_system_message(s, "", true, &ids));
  tpl->render_user_turn(
      py_format_(task, "A rustic, overhead shot of a farm-to-table feast, "
                       "showcasing a variety of colorful, seasonal "
                       "vegetables, artisanal cheeses, and freshly baked "
                       "bread on a weathered wooden table."),
      false, &ids);
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::int32_t> gen{lm->prefill(ctx, ids)};
  const double pre = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(gen[0] >= 0);
  const auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < 6000 && !tpl->is_stop_token(gen.back()); ++i) {
    const std::int32_t t = lm->next_token(ctx);
    if (t < 0) { break; }
    gen.push_back(t);
  }
  const double dec = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t1).count();
  const std::string text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  // The reasoning ends at the template's </think>, which the decode may
  // hand back as vpipe's own marker.
  std::size_t end = std::string::npos, skip = 0;
  for (const std::string_view m : {std::string_view("</think>"),
                                   media_line::kThinkEnd}) {
    const std::size_t at = text.rfind(m);
    if (at != std::string::npos &&
        (end == std::string::npos || at > end)) {
      end = at;
      skip = m.size();
    }
  }
  const std::string answer =
      end == std::string::npos ? text : text.substr(end + skip);
  std::printf("[nvfp4] expander: prompt %zu tokens in %.1f s, %zu written "
              "in %.1f s (%.1f tok/s), thinking %zu chars; answer:\n%s\n",
              ids.size(), pre, gen.size(), dec, gen.size() / dec,
              end == std::string::npos ? 0 : end, answer.c_str());
  EXPECT_TRUE(answer.find("\"expanded_caption\"") != std::string::npos);
  EXPECT_TRUE(answer.find("\"aspect_ratio\"") != std::string::npos);
}
