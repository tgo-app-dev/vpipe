// Qwen-Image-2.1's TEXT ENCODER on this box.
//
// The encoder is a Qwen3-VL 8B, and at bf16 it is 16 GB on disk -- more
// than the DiT and more than a 16 GB machine has. The conditioner loads
// it through the same layer-streaming rule the DiTs use
// (model_memory::plan_streaming -> MetalQwenModel::Config::stream_layers),
// which is exactly the right shape for it: a conditioning encoder only
// ever PREFILLS, so there is no decode to make a streamed layer be read
// twice.
//
// What this checks is that the bf16 checkpoint is reachable at all on
// the box it is run on, and that the tap this family needs comes out of
// it -- the LAST layer's residual, before the final RMSNorm, which is
// the one thing about this encoder that is not shared with every other
// family driving the same weights.

#include "tests/minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/context-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/qwen3/metal-qwen-model.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using vpipe::FlexData;
using vpipe::Session;
using vpipe::genai::ContextId;
using vpipe::genai::ContextManager;
using vpipe::genai::MetalLlamaWeights;
using vpipe::genai::MetalQwenModel;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

// ASSERT_TRUE records and CONTINUES, so anything that would run against
// a failed precondition needs its own guard.
#define Q21C_REQUIRE(cond)                                                 \
  do {                                                                     \
    ASSERT_TRUE(cond);                                                     \
    if (!(cond)) { return; }                                               \
  } while (0)

namespace {

std::string enc_dir_()
{
  const char* p = std::getenv("VPIPE_QWEN_IMAGE21_TEST_MODEL_PATH");
  if (p == nullptr || *p == '\0') { return {}; }
  const std::filesystem::path d =
      std::filesystem::path(p) / "text_encoder";
  std::error_code ec;
  if (!std::filesystem::exists(d / "config.json", ec)) { return {}; }
  return d.string();
}

// The conditioner's own config, kept in step with
// encoder_config_boogu_() -- the same Qwen3-VL 8B under the same
// `model.language_model.` wrapper that Boogu-Image drives.
MetalQwenModel::Config encoder_config_(const std::string& dir)
{
  MetalQwenModel::Config c;
  c.n_layers           = 36;
  c.hidden             = 4096;
  c.n_heads            = 32;
  c.n_kv_heads         = 8;
  c.head_dim           = 128;
  c.rotary_dim         = 128;
  c.ffn_inner          = 12288;
  c.vocab              = 151936;
  c.rope_theta         = 5.0e6f;
  c.rms_eps            = 1e-6f;
  c.full_attn_interval = 1;
  c.tie_embeddings     = false;
  c.use_bf16           = true;
  c.dense              = true;
  c.zero_centered_norm = false;
  c.attn_output_gate   = false;
  c.backbone_only      = true;    // the rows are host-gathered
  c.weight_prefix      = "model.language_model.";
  c.model_seg          = "";
  c.max_seq            = 1024;
  c.page_tokens        = 256;
  std::ifstream in(std::filesystem::path(dir) / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto root = fd.as_object();
      if (root.contains("text_config")) {
        const FlexData tc = root.at("text_config");
        if (tc.is_object()) {
          auto o = tc.as_object();
          auto gi = [&](const char* k, int d) {
            return o.contains(k) ? (int)o.at(k).as_int(d) : d;
          };
          c.n_layers   = gi("num_hidden_layers", c.n_layers);
          c.hidden     = gi("hidden_size", c.hidden);
          c.n_heads    = gi("num_attention_heads", c.n_heads);
          c.n_kv_heads = gi("num_key_value_heads", c.n_kv_heads);
          c.head_dim   = gi("head_dim", c.head_dim);
          c.rotary_dim = c.head_dim;
          c.ffn_inner  = gi("intermediate_size", c.ffn_inner);
          c.vocab      = gi("vocab_size", c.vocab);
        }
      }
    }
  }
  return c;
}

}  // namespace

// STREAMED LAYERS, because 16 GB of bf16 does not sit beside anything.
// Reports what it cost; asserts only that it ran and that the tap is
// finite and non-trivial.
TEST(qwen_image_21_cond, bf16_encoder_streams_and_taps)
{
  const std::string dir = enc_dir_();
  if (dir.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalQwenModel::Config cfg = encoder_config_(dir);
  // Off by default so the suite does not pay it; the streamed arm is
  // what a 16 GB box actually takes.
  const char* nostream = std::getenv("VPIPE_QWEN_IMAGE21_ENC_NO_STREAM");
  cfg.stream_layers = !(nostream != nullptr && *nostream != '\0');

  const auto t_load = std::chrono::steady_clock::now();
  auto m = MetalQwenModel::load(dir, mc, cfg);
  const double load_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t_load).count();
  if (m == nullptr) {
    std::printf("[qwen_image_21_cond] encoder load FAILED (%s layers)\n",
                cfg.stream_layers ? "streamed" : "preloaded");
  }
  Q21C_REQUIRE(m != nullptr);
  std::printf("[qwen_image_21_cond] loaded %d layers in %.0f ms (%s)\n",
              cfg.n_layers, load_ms,
              m->streaming_layers() ? "STREAMED" : "preloaded");

  // Host-gather a short prompt's embedding rows: backbone_only skips
  // the model's own embed muxer, which is what lets the conditioner
  // splice vision tokens into the same stream.
  auto wts = MetalLlamaWeights::open_model(dir);
  Q21C_REQUIRE(wts.has_value());
  SharedBuffer emb =
      wts->load("model.language_model.embed_tokens.weight", mc);
  Q21C_REQUIRE(!emb.empty());

  const int H = cfg.hidden;
  const int n = 48;
  SharedBuffer x = mc->make_shared_buffer((std::size_t)n * H * 2);
  Q21C_REQUIRE(!x.empty());
  const auto* tbl = static_cast<const std::uint8_t*>(emb.contents());
  auto* xb = static_cast<std::uint8_t*>(x.contents());
  for (int i = 0; i < n; ++i) {
    const std::size_t id = (std::size_t)(1000 + i * 37);
    std::memcpy(xb + (std::size_t)i * H * 2, tbl + id * (std::size_t)H * 2,
                (std::size_t)H * 2);
  }
  emb = SharedBuffer{};   // the table is 1.2 GB; do not hold it

  // THE TAP THIS FAMILY NEEDS is the last layer's residual, BEFORE the
  // final RMSNorm -- which is what forward_embeddings_taps returns, and
  // what every other family here then norms on the host.
  ContextManager* cm = m->context_manager();
  const ContextId cid = cm->acquire_root();
  const std::vector<int> taps = {cfg.n_layers - 1};
  const auto t0 = std::chrono::steady_clock::now();
  SharedBuffer out = m->forward_embeddings_taps(cid, x, n, taps);
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  cm->release(cid);
  Q21C_REQUIRE(!out.empty());
  Q21C_REQUIRE(out.byte_size() >= (std::size_t)n * H * 2);

  const auto* d = static_cast<const std::uint16_t*>(out.contents());
  std::size_t bad = 0;
  double sum = 0.0;
  for (std::size_t i = 0; i < (std::size_t)n * H; ++i) {
    std::uint32_t u = (std::uint32_t)d[i] << 16;
    float f;
    std::memcpy(&f, &u, 4);
    if (!std::isfinite(f)) { ++bad; }
    else { sum += (double)f * (double)f; }
  }
  const auto mb = mc->memory_budget();
  std::printf("[qwen_image_21_cond] %d rows prefilled in %.0f ms, tap rms "
              "%.4f, %llu MB compressed\n", n, ms,
              std::sqrt(sum / ((double)n * H)),
              (unsigned long long)(mb.self_compressed >> 20));
  EXPECT_TRUE(bad == 0);
  // A dead tap reads as finite zeros, which is the failure this is for.
  EXPECT_TRUE(sum > 0.0);
}
