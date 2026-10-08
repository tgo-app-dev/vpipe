// FlashVSR's key/value window, held to a latent the forward produced
// before the window's layout changed.
//
// THE REFERENCE IS THIS TREE'S OWN EARLIER OUTPUT, not the model's. The
// question is whether a re-laid-out cache and re-used scratch compute the
// same function, and a saved latent from the layout it replaced answers
// exactly that. It is not bit-exact by construction -- a ring puts the
// keys in a different order, so the attention's float accumulation
// reassociates -- which is why the bound is a rel-L2 and not equality.
//
// FIVE CHUNKS, so the window FILLS and then WRAPS: at kv_ratio 3 the
// fourth window lands in the slot the first trim freed, and at kv_ratio 2
// that happens one chunk earlier. Two runs, one per ratio.
//
// Env, all required or the test skips:
//   VPIPE_FLASHVSR_TEST_MODEL_PATH  a FlashVSR-v1.1 directory (either layout)
//   VPIPE_FVSR_KV_REF               a directory holding kv3.f32 / kv2.f32
//   VPIPE_FVSR_KV_SAVE              set: WRITE the references instead

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"
#include "generative-models/flashvsr/flashvsr-family.h"
#include "generative-models/flashvsr/metal-flashvsr-transformer.h"
#include "generative-models/weight-set.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;

namespace {

namespace fs = std::filesystem;

std::uint16_t
to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

double
rel_l2_(const std::vector<float>& got, const std::vector<float>& ref)
{
  if (got.size() != ref.size() || got.empty()) { return 1e9; }
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double d = (double)got[i] - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * (double)ref[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace

TEST(flashvsr_kv, the_window_matches_a_saved_latent)
{
  const char* root = std::getenv("VPIPE_FLASHVSR_TEST_MODEL_PATH");
  const char* ref = std::getenv("VPIPE_FVSR_KV_REF");
  if (root == nullptr || ref == nullptr || *root == '\0' || *ref == '\0') {
    return;
  }
  const bool save = std::getenv("VPIPE_FVSR_KV_SAVE") != nullptr;
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  FlashVsrLayout layout;
  ASSERT_TRUE(resolve_flashvsr_layout(root, &layout));
  MetalFlashVsrTransformer::Config cfg;
  std::string err;
  ASSERT_TRUE(MetalFlashVsrTransformer::config_from_checkpoint(
      layout.denoiser, &cfg, &err));
  auto ws = WeightSet::open(layout.denoiser, nullptr);
  auto ctx_ws = WeightSet::open(layout.context, nullptr);
  ASSERT_TRUE(ws != nullptr && ctx_ws != nullptr);
  if (ws == nullptr || ctx_ws == nullptr) { return; }
  auto dit = MetalFlashVsrTransformer::load(ws, ctx_ws, layout.context_name,
                                            mc, cfg, &err);
  ASSERT_TRUE(dit != nullptr);
  if (dit == nullptr) {
    std::printf("  load: %s\n", err.c_str());
    return;
  }

  constexpr int kSize = 384;
  // Five chunks by default; VPIPE_FVSR_KV_FRAMES=33 is the two-chunk clip
  // whose window never trims before its last chunk, so the old layout and
  // the ring put every key in the same place and must agree EXACTLY.
  const char* fe = std::getenv("VPIPE_FVSR_KV_FRAMES");
  const int kFrames = (fe != nullptr && std::atoi(fe) > 0) ? std::atoi(fe)
                                                           : 57;
  const int T = MetalFlashVsrTransformer::latent_frames(kFrames);
  const int tok = (kSize / 16) * (kSize / 16);
  const int h8 = kSize / 8;
  std::mt19937_64 rng(0x6b76u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> noise((std::size_t)16 * T * h8 * h8);
  for (auto& v : noise) { v = nd(rng); }
  std::vector<std::uint16_t> rows((std::size_t)T * tok * cfg.hidden);
  for (auto& v : rows) { v = to_bf16_(0.5f * nd(rng)); }

  for (int kv : {3, 2}) {
    MetalFlashVsrTransformer::Request req;
    req.rows = rows.data();
    req.row_frames = T;
    req.frames = kFrames;
    req.height = kSize;
    req.width = kSize;
    req.init_noise = noise.data();
    req.params.kv_ratio = kv;
    std::vector<float> lat;
    std::vector<int> shape;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = dit->generate(req, &lat, &shape, &err);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    ASSERT_TRUE(ok);
    if (!ok) {
      std::printf("  kv_ratio %d: %s\n", kv, err.c_str());
      continue;
    }
    const fs::path f = fs::path(ref) / ("kv" + std::to_string(kv) + ".f32");
    if (save) {
      std::ofstream o(f, std::ios::binary);
      o.write(reinterpret_cast<const char*>(lat.data()),
              (std::streamsize)(lat.size() * sizeof(float)));
      std::printf("  kv_ratio %d: saved %zu values to %s (%.0f ms)\n", kv,
                  lat.size(), f.c_str(), ms);
      continue;
    }
    std::ifstream in(f, std::ios::binary | std::ios::ate);
    ASSERT_TRUE((bool)in);
    if (!in) { continue; }
    std::vector<float> want((std::size_t)in.tellg() / sizeof(float));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(want.data()),
            (std::streamsize)(want.size() * sizeof(float)));
    const double r = rel_l2_(lat, want);
    std::printf("  kv_ratio %d: rel-L2 %.3e against the saved latent "
                "(%.0f ms)\n", kv, r, ms);
    // TWO CHUNKS AT kv_ratio 3 never trim before the last chunk, so the
    // shifting layout and the ring hold every key in the same slot and
    // must agree EXACTLY -- which is the proof the re-used scratch is
    // right. Past that the ring reorders the keys and the attention
    // reassociates. MEASURED at five chunks: 8.8e-3 (kv 3) and 1.0e-2
    // (kv 2), against 1.4e-2 for this same run with ALU attention in
    // place of NAX -- float noise grown through thirty blocks and the
    // cached chunks, not a different function.
    if (kFrames <= 33 && kv >= 3) {
      EXPECT_TRUE(r == 0.0);
    } else {
      EXPECT_TRUE(r < 2e-2);
    }
  }
}

// THE TWO ORDERS COMPUTE ONE FUNCTION. Block-outer runs every chunk
// through a block before the next block, so one key/value window serves
// the whole stack; chunk-outer runs every block of a chunk before the
// next chunk and holds a window per block. A block's work on a chunk
// reads its own residual stream and its own window over the earlier
// chunks, which the reordering leaves identical -- so the latents must
// be EQUAL, bit for bit, not close.
//
// Five chunks so the window fills and WRAPS, at both ratios; and a
// routed arm, because at 384 the default local band covers the whole
// 3x3 window grid and keeps every key -- sparse_ratio 0.05 with
// local_range 1 makes the span lists real. Needs only the checkpoint.
TEST(flashvsr_kv, block_outer_order_is_bit_identical)
{
  const char* root = std::getenv("VPIPE_FLASHVSR_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  FlashVsrLayout layout;
  ASSERT_TRUE(resolve_flashvsr_layout(root, &layout));
  MetalFlashVsrTransformer::Config cfg;
  std::string err;
  ASSERT_TRUE(MetalFlashVsrTransformer::config_from_checkpoint(
      layout.denoiser, &cfg, &err));
  auto ws = WeightSet::open(layout.denoiser, nullptr);
  auto ctx_ws = WeightSet::open(layout.context, nullptr);
  ASSERT_TRUE(ws != nullptr && ctx_ws != nullptr);
  if (ws == nullptr || ctx_ws == nullptr) { return; }
  auto dit = MetalFlashVsrTransformer::load(ws, ctx_ws, layout.context_name,
                                            mc, cfg, &err);
  ASSERT_TRUE(dit != nullptr);
  if (dit == nullptr) { return; }

  constexpr int kSize = 384;
  constexpr int kFrames = 57;   // five chunks: 6 + 4 x 2 latent frames
  const int T = MetalFlashVsrTransformer::latent_frames(kFrames);
  const int tok = (kSize / 16) * (kSize / 16);
  std::mt19937_64 rng(0x6f72u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> noise((std::size_t)16 * T * (kSize / 8) * (kSize / 8));
  for (auto& v : noise) { v = nd(rng); }
  std::vector<std::uint16_t> rows((std::size_t)T * tok * cfg.hidden);
  for (auto& v : rows) { v = to_bf16_(0.5f * nd(rng)); }

  struct Arm { const char* name; double kv, sparse; int local; };
  const Arm arms[] = {{"kv 3", 3.0, 2.0, 11},
                      {"kv 2", 2.0, 2.0, 11},
                      {"kv 3 routed", 3.0, 0.05, 1}};
  for (const Arm& a : arms) {
    MetalFlashVsrTransformer::Request req;
    req.rows = rows.data();
    req.row_frames = T;
    req.frames = kFrames;
    req.height = kSize;
    req.width = kSize;
    req.init_noise = noise.data();
    req.params.kv_ratio = a.kv;
    req.params.sparse_ratio = a.sparse;
    req.params.local_range = a.local;
    // Engagement first: the default order must BE block-outer here, or
    // the comparison below compares a path with itself.
    ::unsetenv("VPIPE_FVSR_CHUNK_OUTER");
    ASSERT_TRUE(MetalFlashVsrTransformer::blocks_outer(
        cfg, kSize, kSize, kFrames, req.params));
    std::vector<float> lat_b, lat_c;
    std::vector<int> shape;
    auto t0 = std::chrono::steady_clock::now();
    const bool ok_b = dit->generate(req, &lat_b, &shape, &err);
    const double ms_b = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
    ASSERT_TRUE(ok_b);
    const std::uint64_t empty_b = dit->empty_query_tiles();
    ::setenv("VPIPE_FVSR_CHUNK_OUTER", "1", 1);
    t0 = std::chrono::steady_clock::now();
    const bool ok_c = dit->generate(req, &lat_c, &shape, &err);
    const double ms_c = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
    ::unsetenv("VPIPE_FVSR_CHUNK_OUTER");
    ASSERT_TRUE(ok_c);
    if (!ok_b || !ok_c) {
      std::printf("  %s: %s\n", a.name, err.c_str());
      continue;
    }
    const bool same = lat_b.size() == lat_c.size() &&
                      std::memcmp(lat_b.data(), lat_c.data(),
                                  lat_b.size() * sizeof(float)) == 0;
    EXPECT_TRUE(same);
    EXPECT_TRUE(dit->empty_query_tiles() == empty_b);
    std::printf("  %-12s block-outer %.0f ms, chunk-outer %.0f ms: %s "
                "(rel-L2 %.2e, %zu values)\n", a.name, ms_b, ms_c,
                same ? "IDENTICAL" : "DIFFERENT", rel_l2_(lat_b, lat_c),
                lat_b.size());
  }

  // SAGE, where it runs at all (the NAX flash entry, so matrix cores):
  // block-outer runs the opening chunk's shape and the steady one inside
  // every block, each against its own Sage object, and must still equal
  // chunk-outer.
  MetalFlashVsrTransformer::Config scfg = cfg;
  scfg.sage.enabled = true;
  auto sdit = MetalFlashVsrTransformer::load(ws, ctx_ws, layout.context_name,
                                             mc, scfg, &err);
  if (sdit != nullptr && sdit->uses_sage()) {
    MetalFlashVsrTransformer::Request req;
    req.rows = rows.data();
    req.row_frames = T;
    req.frames = kFrames;
    req.height = kSize;
    req.width = kSize;
    req.init_noise = noise.data();
    req.params.sparse_ratio = 0.05;
    req.params.local_range = 1;
    std::vector<float> lat_b, lat_c;
    std::vector<int> shape;
    ::unsetenv("VPIPE_FVSR_CHUNK_OUTER");
    ASSERT_TRUE(sdit->generate(req, &lat_b, &shape, &err));
    ::setenv("VPIPE_FVSR_CHUNK_OUTER", "1", 1);
    ASSERT_TRUE(sdit->generate(req, &lat_c, &shape, &err));
    ::unsetenv("VPIPE_FVSR_CHUNK_OUTER");
    const bool same = lat_b.size() == lat_c.size() &&
                      std::memcmp(lat_b.data(), lat_c.data(),
                                  lat_b.size() * sizeof(float)) == 0;
    EXPECT_TRUE(same);
    std::printf("  sage routed  block-outer vs chunk-outer: %s\n",
                same ? "IDENTICAL" : "DIFFERENT");
  } else {
    std::printf("  sage: not available here (needs the NAX flash entry)\n");
  }
}

// THE ORDERS' COST at a real geometry, one arm per process so each
// process's peak footprint is its own: VPIPE_FVSR_ORDER_BENCH=HxWxF
// (e.g. 1152x1920x25) runs one clip in whichever order the environment
// leaves -- VPIPE_FVSR_CHUNK_OUTER=1 for the old one -- and reports the
// generate() time. Wrap the process in a footprint sampler for the peak.
TEST(flashvsr_kv, order_bench)
{
  const char* root = std::getenv("VPIPE_FLASHVSR_TEST_MODEL_PATH");
  const char* spec = std::getenv("VPIPE_FVSR_ORDER_BENCH");
  if (root == nullptr || spec == nullptr || *root == '\0') { return; }
  int H = 0, W = 0, F = 0;
  if (std::sscanf(spec, "%dx%dx%d", &H, &W, &F) != 3) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  FlashVsrLayout layout;
  ASSERT_TRUE(resolve_flashvsr_layout(root, &layout));
  MetalFlashVsrTransformer::Config cfg;
  std::string err;
  ASSERT_TRUE(MetalFlashVsrTransformer::config_from_checkpoint(
      layout.denoiser, &cfg, &err));
  auto ws = WeightSet::open(layout.denoiser, nullptr);
  auto ctx_ws = WeightSet::open(layout.context, nullptr);
  if (ws == nullptr || ctx_ws == nullptr) { return; }
  // VPIPE_FVSR_STREAM=1 streams the blocks, as the family does on a
  // tight box.
  if (const char* e = std::getenv("VPIPE_FVSR_STREAM")) {
    cfg.stream_blocks = *e != '\0' && std::string(e) != "0";
  }
  auto dit = MetalFlashVsrTransformer::load(ws, ctx_ws, layout.context_name,
                                            mc, cfg, &err);
  ASSERT_TRUE(dit != nullptr);
  if (dit == nullptr) { return; }
  F = MetalFlashVsrTransformer::align_frames(F);
  const int T = MetalFlashVsrTransformer::latent_frames(F);
  const std::size_t tok = (std::size_t)(H / 16) * (W / 16);
  std::mt19937_64 rng(0x62u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<std::uint16_t> rows((std::size_t)T * tok * cfg.hidden);
  for (auto& v : rows) { v = to_bf16_(0.5f * nd(rng)); }
  MetalFlashVsrTransformer::Request req;
  req.rows = rows.data();
  req.row_frames = T;
  req.frames = F;
  req.height = H;
  req.width = W;
  const bool outer = MetalFlashVsrTransformer::blocks_outer(cfg, H, W, F,
                                                            req.params);
  std::vector<float> lat;
  std::vector<int> shape;
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = dit->generate(req, &lat, &shape, &err);
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(ok);
  std::printf("[fvsr-order] %dx%d x %d frames, %s%s: generate %.0f ms, "
              "windows %.0f MB, denoise scratch %.0f MB declared\n", W, H, F,
              outer ? "block-outer" : "chunk-outer",
              dit->streams_blocks() ? ", streamed" : "", ms,
              MetalFlashVsrTransformer::kv_window_bytes(cfg, H, W, F,
                                                        req.params) / 1e6,
              MetalFlashVsrTransformer::denoise_scratch_bytes(
                  cfg, H, W, F, req.params) / 1e6);
}

// STREAMED BLOCKS COMPUTE THE RESIDENT FUNCTION, bit for bit. A streaming
// load keeps only what it folds and reads each block's F32 weights into
// one of two slots, narrowing them on the way in -- the same narrowing
// the resident load caches -- so every arm below must EQUAL the resident
// latent: block-outer (the default), chunk-outer (which re-reads the
// stack per chunk), a second clip (the slots reused, block 0 prefetched
// at the end of the clip before), and the per-block-allocation fallback
// VPIPE_FVSR_NO_SLOTS, the A/B that once found a prefetch writing into
// the slot in use. Engagement first: a streaming load that quietly kept
// its weights would pass every comparison.
TEST(flashvsr_kv, streamed_blocks_are_bit_identical)
{
  const char* root = std::getenv("VPIPE_FLASHVSR_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  FlashVsrLayout layout;
  ASSERT_TRUE(resolve_flashvsr_layout(root, &layout));
  MetalFlashVsrTransformer::Config cfg;
  std::string err;
  ASSERT_TRUE(MetalFlashVsrTransformer::config_from_checkpoint(
      layout.denoiser, &cfg, &err));
  auto ws = WeightSet::open(layout.denoiser, nullptr);
  auto ctx_ws = WeightSet::open(layout.context, nullptr);
  if (ws == nullptr || ctx_ws == nullptr) { return; }
  auto load = [&](bool stream) {
    MetalFlashVsrTransformer::Config c = cfg;
    c.stream_blocks = stream;
    return MetalFlashVsrTransformer::load(ws, ctx_ws, layout.context_name, mc,
                                          c, &err);
  };

  constexpr int kSize = 384;
  constexpr int kFrames = 57;
  const int T = MetalFlashVsrTransformer::latent_frames(kFrames);
  const int tok = (kSize / 16) * (kSize / 16);
  std::mt19937_64 rng(0x73u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> noise((std::size_t)16 * T * (kSize / 8) * (kSize / 8));
  for (auto& v : noise) { v = nd(rng); }
  std::vector<std::uint16_t> rows((std::size_t)T * tok * cfg.hidden);
  for (auto& v : rows) { v = to_bf16_(0.5f * nd(rng)); }
  MetalFlashVsrTransformer::Request req;
  req.rows = rows.data();
  req.row_frames = T;
  req.frames = kFrames;
  req.height = kSize;
  req.width = kSize;
  req.init_noise = noise.data();
  req.params.sparse_ratio = 0.05;   // routed, so the span lists are real
  req.params.local_range = 1;
  ::unsetenv("VPIPE_FVSR_CHUNK_OUTER");
  ::unsetenv("VPIPE_FVSR_NO_SLOTS");

  std::vector<float> ref;
  std::vector<int> shape;
  std::uint64_t held_resident = 0;
  {
    auto dit = load(false);
    ASSERT_TRUE(dit != nullptr);
    if (dit == nullptr) { return; }
    EXPECT_FALSE(dit->streams_blocks());
    held_resident = dit->resident_bytes();
    ASSERT_TRUE(dit->generate(req, &ref, &shape, &err));
  }
  auto same = [&](const std::vector<float>& got) {
    return got.size() == ref.size() &&
           std::memcmp(got.data(), ref.data(), got.size() * 4) == 0;
  };

  auto dit = load(true);
  ASSERT_TRUE(dit != nullptr);
  if (dit == nullptr) { return; }
  ASSERT_TRUE(dit->streams_blocks());
  std::vector<float> lat;
  ASSERT_TRUE(dit->generate(req, &lat, &shape, &err));
  const int hits1 = dit->prefetch_hits();
  const std::uint64_t held = dit->resident_bytes();
  EXPECT_TRUE(same(lat));
  // Every block but the first was read under the one before it.
  EXPECT_TRUE(hits1 >= cfg.n_layers - 1);
  // The weights left; the folds and the slot pair are what is held.
  EXPECT_TRUE(held > 0 && held * 4 < held_resident);
  EXPECT_TRUE(held <= MetalFlashVsrTransformer::streaming_floor_bytes(cfg));
  std::printf("  streamed, block-outer: %s; %d of %d blocks prefetched; holds "
              "%.0f MB against %.0f MB resident (floor declared %.0f MB)\n",
              same(lat) ? "IDENTICAL" : "DIFFERENT", hits1, cfg.n_layers,
              held / 1e6, held_resident / 1e6,
              MetalFlashVsrTransformer::streaming_floor_bytes(cfg) / 1e6);

  // A second clip on the same slots: block 0 was read ahead at the end
  // of the first, so this one is prefetched throughout.
  ASSERT_TRUE(dit->generate(req, &lat, &shape, &err));
  EXPECT_TRUE(same(lat));
  EXPECT_TRUE(dit->prefetch_hits() == cfg.n_layers);
  std::printf("  streamed, second clip: %s; %d of %d prefetched\n",
              same(lat) ? "IDENTICAL" : "DIFFERENT", dit->prefetch_hits(),
              cfg.n_layers);

  ::setenv("VPIPE_FVSR_CHUNK_OUTER", "1", 1);
  ASSERT_TRUE(dit->generate(req, &lat, &shape, &err));
  ::unsetenv("VPIPE_FVSR_CHUNK_OUTER");
  EXPECT_TRUE(same(lat));
  std::printf("  streamed, chunk-outer: %s\n",
              same(lat) ? "IDENTICAL" : "DIFFERENT");

  // The fallback: a block built per acquire, no pair, no read ahead. Read
  // once by the slots, on their first acquire -- so a fresh load.
  ::setenv("VPIPE_FVSR_NO_SLOTS", "1", 1);
  {
    auto d2 = load(true);
    ASSERT_TRUE(d2 != nullptr);
    if (d2 != nullptr) {
      ASSERT_TRUE(d2->generate(req, &lat, &shape, &err));
      EXPECT_TRUE(same(lat));
      EXPECT_TRUE(d2->prefetch_hits() == 0);
      std::printf("  streamed, VPIPE_FVSR_NO_SLOTS: %s\n",
                  same(lat) ? "IDENTICAL" : "DIFFERENT");
    }
  }
  ::unsetenv("VPIPE_FVSR_NO_SLOTS");
}
