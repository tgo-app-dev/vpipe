// MiniMax-Music3 bring-up tests: the prompt contract, the AR stage, the
// flow matching and the vocoder, each against the diffusers reference's
// own output (tools/dump_minimax_music3_golden.py).
//
// Env:
//   VPIPE_MUSIC3_TEST_MODEL_PATH  the MiniMaxAI/MiniMax-Music3 directory
//                                 (its diffusers layout).
//   VPIPE_MUSIC3_GOLDEN           the golden directory: prompt_cases.json
//                                 and ar/, ar_bf16/, dit/, dit_bf16/, voc/.
// Every test that needs either skips without it; protocol_logic needs
// neither.
//
// THE BAR is the reference's own bf16 run against its fp32 one, at the
// same inputs (the *_bf16 goldens are teacher-forced or fed the same
// noise).
//
// GEOMETRY. The goldens are a 250-frame song: two windows (200 frames ->
// 689 latents, then 150 -> 516, the second blended onto the first's last
// 172), so the DiT runs 1380- and 1034-row GEMMs (both guidance rows) and
// an attention of 690 / 517 keys, neither a multiple of any tile.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-transcribe-stage.h"
#include "stages/audio-vae-decode-stage.h"
#include "stages/audio-video/audio-temporal-resample-stage.h"
#include "stages/generate-audio-stage.h"
#include "stages/model-detect.h"
#include "stages/model-memory.h"
#include "generative-models/minimax-music3/metal-music3-ar.h"
#include "generative-models/minimax-music3/metal-music3-dit.h"
#include "generative-models/minimax-music3/music3-protocol.h"
#include "generative-models/shared/torch-randn.h"
#include "generative-models/tokenizer.h"
#include "generative-models/weight-set.h"
#include "generative-models/yue2/metal-oobleck-decoder.h"

#include <mach/mach.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
namespace fs = std::filesystem;

namespace {

std::string
env_(const char* k)
{
  const char* v = std::getenv(k);
  return (v != nullptr && *v != '\0') ? std::string(v) : std::string();
}

FlexData
read_json_(const fs::path& p)
{
  std::ifstream in(p);
  if (!in) { return FlexData(); }
  return FlexData::from_json(in);
}

// One array of a golden directory: <name>.bin with its shape in meta.json.
template <typename T>
std::vector<T>
golden_(const fs::path& dir, const std::string& name,
        std::vector<int>* shape = nullptr)
{
  std::vector<T> out;
  const FlexData meta = read_json_(dir / "meta.json");
  if (!meta.is_object() || !meta.as_object().contains(name)) { return out; }
  const FlexData m = meta.as_object().at(name);
  std::size_t n = 1;
  if (shape != nullptr) { shape->clear(); }
  const FlexData sh = m.as_object().at("shape");
  for (const FlexData& d : sh.as_array()) {
    n *= (std::size_t)d.as_int();
    if (shape != nullptr) { shape->push_back((int)d.as_int()); }
  }
  std::ifstream in(dir / (name + ".bin"), std::ios::binary);
  if (!in) { return out; }
  out.resize(n);
  in.read(reinterpret_cast<char*>(out.data()),
          (std::streamsize)(n * sizeof(T)));
  return out;
}

double
rel_l2_(const float* a, const float* b, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// [rows, cols] -> [cols, rows].
std::vector<float>
transpose_(const std::vector<float>& a, int rows, int cols)
{
  std::vector<float> o(a.size());
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      o[(std::size_t)c * rows + r] = a[(std::size_t)r * cols + c];
    }
  }
  return o;
}

}  // namespace

// ---- the protocol, no model ---------------------------------------------

TEST(minimax_music3, protocol_logic)
{
  using namespace music3;
  // Windows: one up to 200 frames, then range(0, n - 100, 100).
  EXPECT_TRUE(window_starts(1) == std::vector<int>({0}));
  EXPECT_TRUE(window_starts(200) == std::vector<int>({0}));
  EXPECT_TRUE(window_starts(201) == std::vector<int>({0, 100}));
  EXPECT_TRUE(window_starts(250) == std::vector<int>({0, 100}));
  EXPECT_TRUE(window_starts(301) == std::vector<int>({0, 100, 200}));
  EXPECT_TRUE(window_starts(4500).size() == 44);
  // int(n * 44100 / 24000 * 960 / 512).
  EXPECT_TRUE(latent_length(200) == 689);
  EXPECT_TRUE(latent_length(150) == 516);
  EXPECT_TRUE(latent_length(1) == 3);
  // The stitched song: the first window to L - 258, the rest from 86.
  EXPECT_TRUE(stitched_latents(250) == (689 - 258) + (516 - 86));
  EXPECT_TRUE(stitched_latents(200) == 689);
  EXPECT_TRUE(kept_latents(0, 1, 689) == std::make_pair(0, 689));
  EXPECT_TRUE(kept_latents(1, 3, 689) == std::make_pair(86, 431));
  // The kept seam is inside the 172 latents the next window copies: the
  // first window keeps to L - 258 = 431, the copy covers [L - 344, L -
  // 172) = [345, 517), and 431 - 345 is the next window's 86.
  EXPECT_TRUE(689 - kCropRight - (689 - 2 * kOverlapLatents) == kCropLeft);
  // torch's nearest: 200 frames to 689 latents.
  const std::vector<int> idx = nearest_indices(200, 689);
  EXPECT_TRUE(idx.front() == 0 && idx.back() == 199);
  for (std::size_t i = 1; i < idx.size(); ++i) {
    EXPECT_TRUE(idx[i] >= idx[i - 1] && idx[i] - idx[i - 1] <= 1);
  }
  // The flow grid: 0 first, 1 last, increasing.
  const std::vector<float> t = flow_times(30);
  EXPECT_TRUE(t.size() == 31 && t[0] == 0.0f && t[30] == 1.0f);
  for (std::size_t i = 1; i < t.size(); ++i) { EXPECT_TRUE(t[i] > t[i - 1]); }
  EXPECT_TRUE(flow_times(1) == std::vector<float>({0.0f, 1.0f}));
  // The guidance branch keeps the first id and the last two.
  const std::vector<std::int32_t> ids = {1, 2, 3, 4, 5};
  EXPECT_TRUE(unconditional_ids(ids) ==
              std::vector<std::int32_t>({1, kAudioCfg, kAudioCfg, 4, 5}));
  // Unicode the cleaners rely on.
  EXPECT_TRUE(unicode_lower("[ПРИПЕВ] Été ÜBER") ==
              "[припев] été über");
  EXPECT_TRUE(py_isspace(0x2028) && py_isspace(0xa0) && py_isspace(0x1f) &&
              !py_isspace(0x200b));

  // The picks: a dominant code wins, the barred ids never.
  std::vector<float> c((std::size_t)kHeadRows, -10.0f);
  std::vector<float> u((std::size_t)kHeadRows, -10.0f);
  c[(std::size_t)kSemanticBase + 7] = 30.0f;
  c[1] = 100.0f;                      // <|caption_start|>: never a pick
  FramePicker p(1);
  for (int i = 0; i < 20; ++i) {
    EXPECT_TRUE(p.pick_semantic(c.data(), u.data()) == kSemanticBase + 7);
  }
  // Guidance moves the pick: the conditional row prefers code 3, but the
  // unconditional row likes code 3 even more, so guidance prefers 4.
  std::vector<float> rc((std::size_t)kResidualCodes, -30.0f);
  std::vector<float> ru((std::size_t)kResidualCodes, -30.0f);
  rc[3] = 10.0f; rc[4] = 9.0f;
  ru[3] = 30.0f; ru[4] = 0.0f;
  std::vector<float> sc;
  FramePicker::residual_scores(rc.data(), ru.data(), kResidualCodes, &sc);
  EXPECT_TRUE(sc[4] > sc[3]);
}

// ---- the prompt contract ---------------------------------------------------

TEST(minimax_music3, prompt_matches_reference)
{
  const std::string mp = env_("VPIPE_MUSIC3_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_MUSIC3_GOLDEN");
  if (mp.empty() || gp.empty()) { return; }
  const fs::path tj = fs::path(mp) / "tokenizer" / "tokenizer.json";
  auto tok = Tokenizer::from_huggingface_json(tj.string(), nullptr);
  ASSERT_TRUE(tok != nullptr);
  if (tok == nullptr) { return; }
  music3::AddedTokens added;
  std::string err;
  ASSERT_TRUE(music3::read_added_tokens(tj.string(), &added, &err));
  EXPECT_TRUE(added.size() == 32);
  const FlexData cases = read_json_(fs::path(gp) / "prompt_cases.json");
  ASSERT_TRUE(cases.is_array());
  if (!cases.is_array()) { return; }
  int n = 0;
  for (const FlexData& c : cases.as_array()) {
    auto o = c.as_object();
    const std::string cap(o.at("caption").as_string());
    const std::string lyr(o.at("lyrics").as_string());
    const std::string want(o.at("text").as_string());
    const std::string got = music3::prompt_text(cap, lyr);
    if (got != want) {
      std::printf("[music3] case %d text differs:\n  got  %s\n  want %s\n",
                  n, got.c_str(), want.c_str());
    }
    EXPECT_TRUE(got == want);
    std::vector<std::int32_t> ids_want;
    const FlexData idv = o.at("ids");
    for (const FlexData& v : idv.as_array()) {
      ids_want.push_back((std::int32_t)v.as_int());
    }
    const auto ids = music3::encode_text(*tok, added, want);
    if (ids != ids_want) {
      std::size_t i = 0;
      while (i < ids.size() && i < ids_want.size() && ids[i] == ids_want[i]) {
        ++i;
      }
      std::printf("[music3] case %d ids differ at %zu (%zu vs %zu)\n", n, i,
                  ids.size(), ids_want.size());
    }
    EXPECT_TRUE(ids == ids_want);
    ++n;
  }
  std::printf("[music3] %d prompt cases\n", n);
  EXPECT_TRUE(n >= 4);
}

// ---- the vocoder ----------------------------------------------------------

TEST(minimax_music3, vocoder_matches_reference)
{
  const std::string mp = env_("VPIPE_MUSIC3_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_MUSIC3_GOLDEN");
  if (mp.empty() || gp.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string vdir = (fs::path(mp) / "vocoder").string();
  EXPECT_TRUE(MetalOobleckDecoder::is_music3_vocoder_dir(vdir));
  EXPECT_TRUE(!MetalOobleckDecoder::is_oobleck_dir(vdir));
  std::string err;
  auto voc = MetalOobleckDecoder::load(vdir, mc, &err);
  if (voc == nullptr) { std::printf("[music3] vocoder: %s\n", err.c_str()); }
  ASSERT_TRUE(voc != nullptr);
  if (voc == nullptr) { return; }
  EXPECT_TRUE(voc->config().hop() == 512 &&
              voc->config().sample_rate == 44100 &&
              voc->config().streams == 2);
  EXPECT_TRUE(voc->natural_length(100) == 51200);
  const fs::path g = fs::path(gp) / "voc";
  std::vector<int> ls, ws;
  const auto lat = golden_<float>(g, "latent", &ls);       // [128, T]
  const auto want = golden_<float>(g, "wave", &ws);        // [2, 512 T]
  ASSERT_TRUE(ls.size() == 2 && ls[0] == 128 && !want.empty());
  if (ls.size() != 2 || want.empty()) { return; }
  const int T = ls[1];
  // Whole, then 64-latent tiles: the halo must make the seams exact.
  std::vector<float> first;
  for (int core : {1024, 64}) {
    voc->set_core_frames(core);
    std::vector<float> pcm;
    ASSERT_TRUE(voc->decode(lat.data(), T, &pcm, /*clamp=*/true, {}, &err));
    ASSERT_TRUE(pcm.size() == want.size());
    if (pcm.size() != want.size()) { return; }
    const double rl = rel_l2_(pcm.data(), want.data(), pcm.size());
    double mx = 0.0;
    for (std::size_t i = 0; i < pcm.size(); ++i) {
      mx = std::max(mx, (double)std::fabs(pcm[i] - want[i]));
    }
    std::printf("[music3] vocoder core %d: rel-L2 %.3e, max |d| %.3e "
                "(%d latents, the reference's per-window crops)\n", core, rl,
                mx, T);
    EXPECT_TRUE(rl < 5e-3);
    if (first.empty()) {
      first = pcm;
    } else {
      // Tiling is a memory knob: with the halo the tiles see what the
      // whole decode sees. Bit-identical on steel (M4); on matrix cores a
      // GEMM's route follows its row count, so a tile's GEMMs round
      // differently from the whole's. MEASURED on the base M5: 5.5e-4
      // apart, each 6.3e-4 from the reference -- the f16 activations'
      // rounding, whichever route takes it.
      const double dt = rel_l2_(pcm.data(), first.data(), pcm.size());
      std::printf("[music3] vocoder tiled vs whole: %.3e\n", dt);
      EXPECT_TRUE(dt < (voc->uses_mma() ? 1e-3 : 1e-6));
    }
  }
}

// ---- the flow matching ----------------------------------------------------

TEST(minimax_music3, dit_matches_reference)
{
  const std::string mp = env_("VPIPE_MUSIC3_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_MUSIC3_GOLDEN");
  if (mp.empty() || gp.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMusic3Dit::Config cfg;
  std::string err;
  ASSERT_TRUE(MetalMusic3Dit::config_from_dir(mp, &cfg, &err));
  auto dit = MetalMusic3Dit::load(
      WeightSet::open((fs::path(mp) / "transformer").string(), nullptr),
      WeightSet::open((fs::path(mp) / "condition_encoder").string(),
                      nullptr),
      mc, cfg, &err);
  if (dit == nullptr) { std::printf("[music3] dit: %s\n", err.c_str()); }
  ASSERT_TRUE(dit != nullptr);
  if (dit == nullptr) { return; }
  std::printf("[music3] DiT on %s GEMMs, %s attention, %.2f GB\n",
              dit->uses_mma() ? "matmul2d" : "steel",
              dit->uses_attn_nax() ? "NAX" : "steel",
              (double)dit->resident_bytes() / 1e9);
  EXPECT_TRUE(dit->uses_mma() == (mc->supports_matrix_cores() &&
                                  std::getenv("VPIPE_MUSIC3_NO_MMA2") ==
                                      nullptr));
  const fs::path ar = fs::path(gp) / "ar";
  const fs::path g = fs::path(gp) / "dit";
  const fs::path gb = fs::path(gp) / "dit_bf16";
  std::vector<int> hs;
  const auto hid32 = golden_<float>(ar, "hiddens", &hs);   // [250, 32768]
  ASSERT_TRUE(hs.size() == 2 && hs[1] == 32768);
  if (hs.size() != 2) { return; }
  const int F = hs[0];
  std::vector<_Float16> hid(hid32.size());
  for (std::size_t i = 0; i < hid.size(); ++i) {
    hid[i] = (_Float16)hid32[i];
  }
  const std::vector<int> starts = music3::window_starts(F);
  ASSERT_TRUE(starts.size() == 2);

  // 1. The condition of each window (before the overlap splice).
  for (int k = 0; k < 1; ++k) {
    const int s0 = starts[(std::size_t)k];
    const int s1 = std::min(s0 + music3::kChunkFrames, F);
    std::vector<float> cond;
    ASSERT_TRUE(dit->condition(hid.data() + (std::size_t)s0 * 32768, s1 - s0,
                               &cond, &err));
    const auto want = golden_<float>(g, "w" + std::to_string(k) + "_cond");
    const auto bar = golden_<float>(gb, "w" + std::to_string(k) + "_cond");
    ASSERT_TRUE(cond.size() == want.size());
    if (cond.size() != want.size()) { return; }
    const double d = rel_l2_(cond.data(), want.data(), want.size());
    const double b = rel_l2_(bar.data(), want.data(), want.size());
    std::printf("[music3] window %d condition: %.4f from fp32 (ref bf16 "
                "%.4f)\n", k, d, b);
    EXPECT_TRUE(d < b);
  }

  // 2. Single evaluations at the fp32 trajectory's own states.
  {
    const int L = 689, C = 128;
    const auto xs = golden_<float>(g, "w0_x");          // [30, 128, L]
    const auto vcs = golden_<float>(g, "w0_vc");
    const auto vus = golden_<float>(g, "w0_vu");
    const auto ts = golden_<float>(g, "w0_t");
    const auto cond = golden_<float>(g, "w0_cond");     // [L, 2048]
    const auto vcb = golden_<float>(gb, "w0_vc");
    const std::size_t n = (std::size_t)C * L;
    // The Euler grid is the scheduler's.
    const std::vector<float> grid = music3::flow_times(30);
    for (int i = 0; i < 30; ++i) {
      EXPECT_TRUE(grid[(std::size_t)i] == ts[(std::size_t)i]);
    }
    for (int i : {0, 15, 29}) {
      const std::vector<float> x0(xs.begin() + (std::ptrdiff_t)(i * n),
                                  xs.begin() + (std::ptrdiff_t)((i + 1) * n));
      std::vector<float> vc, vu;
      ASSERT_TRUE(dit->velocity(transpose_(x0, C, L), ts[(std::size_t)i],
                                cond, &vc, &vu, &err));
      const auto gvc = transpose_(std::vector<float>(
          vcs.begin() + (std::ptrdiff_t)(i * n),
          vcs.begin() + (std::ptrdiff_t)((i + 1) * n)), C, L);
      const auto gvu = transpose_(std::vector<float>(
          vus.begin() + (std::ptrdiff_t)(i * n),
          vus.begin() + (std::ptrdiff_t)((i + 1) * n)), C, L);
      const double dc = rel_l2_(vc.data(), gvc.data(), n);
      const double du = rel_l2_(vu.data(), gvu.data(), n);
      std::printf("[music3] step %2d velocity: cond %.4f uncond %.4f from "
                  "fp32\n", i, dc, du);
      if (i == 0) {
        // Step 0 is the one state both references share (the noise), so
        // the reference bf16's distance there is a teacher-forced bar.
        const double b = rel_l2_(vcb.data(), vcs.data(), n);
        std::printf("[music3]          (ref bf16 at step 0: %.4f)\n", b);
        EXPECT_TRUE(dc < b && du < b);
      } else {
        EXPECT_TRUE(dc < 0.0165 && du < 0.0165);
      }
    }
  }

  // 3. The whole song from the golden's noise, both windows, stitched.
  {
    std::vector<std::vector<float>> noise;
    for (int k = 0; k < 2; ++k) {
      noise.push_back(golden_<float>(g, "w" + std::to_string(k) + "_noise"));
    }
    std::vector<float> lat;
    const auto t0 = std::chrono::steady_clock::now();
    ASSERT_TRUE(dit->synthesize(hid.data(), F, 7, 30, 1.7f, &lat, {}, &err,
                                &noise));
    const double s = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0).count();
    std::printf("[music3] 2 windows x 30 steps in %.1f s (%.0f ms an "
                "evaluation of both rows)\n", s,
                dit->stats().ms / std::max(1, dit->stats().evals));
    const int T = (int)(lat.size() / 128);
    // The golden's per-window latents, stitched the same way.
    std::vector<float> want((std::size_t)128 * T), bar(want.size());
    int o = 0;
    for (int k = 0; k < 2; ++k) {
      std::vector<int> sh;
      const auto w = golden_<float>(g, "w" + std::to_string(k) + "_latent",
                                    &sh);
      const auto wb = golden_<float>(gb, "w" + std::to_string(k) + "_latent");
      const int L = sh[1];
      const auto kept = music3::kept_latents(k, 2, L);
      for (int j = kept.first; j < kept.second; ++j, ++o) {
        for (int c = 0; c < 128; ++c) {
          want[(std::size_t)c * T + o] = w[(std::size_t)c * L + j];
          bar[(std::size_t)c * T + o] = wb[(std::size_t)c * L + j];
        }
      }
    }
    ASSERT_TRUE(o == T);
    const double d = rel_l2_(lat.data(), want.data(), want.size());
    const double b = rel_l2_(bar.data(), want.data(), want.size());
    std::printf("[music3] stitched latents: %.4f from fp32 (ref bf16 %.4f)\n",
                d, b);
    EXPECT_TRUE(d < b);
    // And the noise synthesize() draws for a seed is the golden's: torch's
    // CPU generator, one float32 draw a window, in order.
    TorchCpuRandn gen(7);
    for (int k = 0; k < 2; ++k) {
      const auto z = gen.randn(noise[(std::size_t)k].size());
      EXPECT_TRUE(z == noise[(std::size_t)k]);
    }
  }
}

// The blocks' int8 tiers (matrix cores only: the int8 context declines
// elsewhere, and this says so and returns). Each tier ENGAGED -- all 144
// block projections of an evaluation on the int8 route at a full window,
// and a held layout DEQUANTIZED under the route's floor at a short one --
// HELD as booked (an int8 layout and no f16 copy beside it), one
// evaluation at the fp32 trajectory's states within a bar of the f16 run,
// the 2-window song from the golden's noise, and a window's device peak
// within its booking.
TEST(minimax_music3, int8_tiers_hold_their_layout)
{
  const std::string mp = env_("VPIPE_MUSIC3_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_MUSIC3_GOLDEN");
  if (mp.empty() || gp.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[music3] no matrix cores: the int8 tiers decline here\n");
    return;
  }
  MetalMusic3Dit::Config base;
  std::string err;
  ASSERT_TRUE(MetalMusic3Dit::config_from_dir(mp, &base, &err));
  const fs::path g = fs::path(gp) / "dit";
  const fs::path gb = fs::path(gp) / "dit_bf16";
  std::vector<int> hs;
  const auto hid32 = golden_<float>(fs::path(gp) / "ar", "hiddens", &hs);
  ASSERT_TRUE(hs.size() == 2 && hs[1] == 32768);
  if (hs.size() != 2) { return; }
  const int F = hs[0];
  std::vector<_Float16> hid(hid32.size());
  for (std::size_t i = 0; i < hid.size(); ++i) {
    hid[i] = (_Float16)hid32[i];
  }
  const int L = 689, C = 128, CD = 2048;
  const std::size_t n = (std::size_t)C * L;
  const auto xs = golden_<float>(g, "w0_x");
  const auto vcs = golden_<float>(g, "w0_vc");
  const auto ts = golden_<float>(g, "w0_t");
  const auto cond = golden_<float>(g, "w0_cond");
  const auto vcb = golden_<float>(gb, "w0_vc");
  const double bar0 = rel_l2_(vcb.data(), vcs.data(), n);
  std::vector<std::vector<float>> noise;
  // The reference's own bf16 song, stitched as ours is: the song's bar.
  std::vector<float> want_song, bar_song;
  {
    std::vector<int> sh0, sh1;
    const auto w0 = golden_<float>(g, "w0_latent", &sh0);
    const auto w1 = golden_<float>(g, "w1_latent", &sh1);
    const auto b0 = golden_<float>(gb, "w0_latent");
    const auto b1 = golden_<float>(gb, "w1_latent");
    const std::vector<float>* ws[2] = {&w0, &w1};
    const std::vector<float>* bs[2] = {&b0, &b1};
    const int Ls[2] = {sh0[1], sh1[1]};
    const auto k0 = music3::kept_latents(0, 2, Ls[0]);
    const auto k1 = music3::kept_latents(1, 2, Ls[1]);
    const int T = (k0.second - k0.first) + (k1.second - k1.first);
    want_song.assign((std::size_t)128 * T, 0.0f);
    bar_song.assign(want_song.size(), 0.0f);
    int o = 0;
    for (int k = 0; k < 2; ++k) {
      const auto kept = music3::kept_latents(k, 2, Ls[k]);
      for (int j = kept.first; j < kept.second; ++j, ++o) {
        for (int c = 0; c < 128; ++c) {
          want_song[(std::size_t)c * T + o] =
              (*ws[k])[(std::size_t)c * Ls[k] + j];
          bar_song[(std::size_t)c * T + o] =
              (*bs[k])[(std::size_t)c * Ls[k] + j];
        }
      }
    }
  }
  const double bar_s =
      rel_l2_(bar_song.data(), want_song.data(), want_song.size());
  for (int k = 0; k < 2; ++k) {
    noise.push_back(golden_<float>(g, "w" + std::to_string(k) + "_noise"));
  }
  // A short window, under the int8 route's 1024-row floor: 347 latents
  // (2 x 348 rows) of step 15's state and the window's condition.
  const int Ls = 347;
  const std::vector<float> x15 = transpose_(
      std::vector<float>(xs.begin() + (std::ptrdiff_t)(15 * n),
                         xs.begin() + (std::ptrdiff_t)(16 * n)), C, L);
  const std::vector<float> xs_short(x15.begin(),
                                    x15.begin() + (std::ptrdiff_t)Ls * C);
  const std::vector<float> cs_short(cond.begin(),
                                    cond.begin() + (std::ptrdiff_t)Ls * CD);
  // ... and a tiny one, 8 latents (18 rows): one partial tile.
  const int Lt = 8;
  const std::vector<float> xs_tiny(x15.begin(),
                                   x15.begin() + (std::ptrdiff_t)Lt * C);
  const std::vector<float> cs_tiny(cond.begin(),
                                   cond.begin() + (std::ptrdiff_t)Lt * CD);
  // Where a held layout meets the f16 route: under a floor set from the
  // environment (the engine lowers it to every row for a held layout).
  const bool env_floor = std::getenv("VPIPE_I8_GEMM_MIN_M") != nullptr;

  std::vector<float> f16_short, f16_eval, f16_tiny;
  for (int tier : {-1, 0, 1, 2, 3}) {
    MetalMusic3Dit::Config cfg = base;
    cfg.i8_tier = tier;
    const auto tl0 = std::chrono::steady_clock::now();
    auto dit = MetalMusic3Dit::load(
        WeightSet::open((fs::path(mp) / "transformer").string(), nullptr),
        WeightSet::open((fs::path(mp) / "condition_encoder").string(),
                        nullptr),
        mc, cfg, &err);
    const double load_s = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - tl0).count();
    ASSERT_TRUE(dit != nullptr);
    if (dit == nullptr) { return; }
    const int eff = MetalMusic3Dit::effective_i8_tier(tier, true);
    EXPECT_TRUE(dit->i8_tier() == eff);
    // Held == booked, to the byte.
    const std::size_t held = dit->resident_bytes() - dit->condition_bytes();
    const std::size_t booked = MetalMusic3Dit::transformer_weight_bytes(mp,
                                                                        eff);
    EXPECT_TRUE(held == booked);
    // The device's peak over the evaluations and the song, after the load.
    const std::size_t alloc0 = mc->memory_budget().allocated;
    std::atomic<std::size_t> peak{alloc0};
    std::atomic<bool> done{false};
    std::thread sampler([&] {
      while (!done.load()) {
        const std::size_t a = mc->memory_budget().allocated;
        if (a > peak.load()) { peak.store(a); }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
    });
    // 1. Evaluations at the fp32 trajectory's states.
    double worst = 0.0, at0 = 0.0;
    long long per_eval = 0;
    for (int i : {0, 15, 29}) {
      const std::vector<float> x0 = transpose_(
          std::vector<float>(xs.begin() + (std::ptrdiff_t)(i * n),
                             xs.begin() + (std::ptrdiff_t)((i + 1) * n)),
          C, L);
      std::vector<float> vc, vu;
      const long long g0 = dit->i8_gemms();
      ASSERT_TRUE(dit->velocity(x0, ts[(std::size_t)i], cond, &vc, &vu,
                                &err));
      per_eval = dit->i8_gemms() - g0;
      const auto gvc = transpose_(std::vector<float>(
          vcs.begin() + (std::ptrdiff_t)(i * n),
          vcs.begin() + (std::ptrdiff_t)((i + 1) * n)), C, L);
      const double d = rel_l2_(vc.data(), gvc.data(), n);
      worst = std::max(worst, d);
      if (i == 0) { at0 = d; }
      if (i == 15 && tier < 0) { f16_eval = vc; }
      if (i == 15 && tier >= 0 && !f16_eval.empty()) {
        std::printf("[music3]   %s step 15: %.4f from the f16 run\n",
                    MetalMusic3Dit::i8_tier_name(eff),
                    rel_l2_(vc.data(), f16_eval.data(), n));
      }
    }
    // 2. The short window: the held layouts dequantized, the f16 route.
    std::vector<float> svc, svu;
    const long long d0 = dit->i8_dequant_gemms();
    const long long s0 = dit->i8_gemms();
    ASSERT_TRUE(dit->velocity(xs_short, ts[15], cs_short, &svc, &svu, &err));
    const long long deq = dit->i8_dequant_gemms() - d0;
    const long long short_i8 = dit->i8_gemms() - s0;
    std::vector<float> tvc, tvu;
    ASSERT_TRUE(dit->velocity(xs_tiny, ts[15], cs_tiny, &tvc, &tvu, &err));
    double tiny_d = 0.0;
    if (tier < 0) {
      f16_tiny = tvc;
    } else if (!f16_tiny.empty()) {
      tiny_d = rel_l2_(tvc.data(), f16_tiny.data(), f16_tiny.size());
    }
    // ... and its time, warm (VPIPE_I8_GEMM_MIN_M moves the floor).
    const auto ts0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 3; ++r) {
      ASSERT_TRUE(dit->velocity(xs_short, ts[15], cs_short, &svc, &svu,
                                &err));
    }
    const double short_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - ts0)
                                .count() / 3.0;
    double short_d = 0.0;
    if (tier < 0) {
      f16_short = svc;
    } else if (!f16_short.empty()) {
      short_d = rel_l2_(svc.data(), f16_short.data(), f16_short.size());
    }
    // 3. The song.
    std::vector<float> lat;
    ASSERT_TRUE(dit->synthesize(hid.data(), F, 7, 30, 1.7f, &lat, {}, &err,
                                &noise));
    done.store(true);
    sampler.join();
    const double ms = dit->stats().ms / std::max(1, dit->stats().evals);
    ASSERT_TRUE(lat.size() == want_song.size());
    if (lat.size() != want_song.size()) { return; }
    const double song = rel_l2_(lat.data(), want_song.data(), lat.size());
    const std::size_t used = peak.load() - alloc0;
    const std::size_t wbook = MetalMusic3Dit::window_scratch_bytes(cfg, eff);
    std::printf("[music3] %-11s held %7.1f MB (booked %7.1f), eval %.4f "
                "(step 0 %.4f; ref bf16 %.4f), short window %.4f from f16 "
                "(%lld dequantized), song %.4f, %lld int8 GEMMs an eval, "
                "%.0f ms an eval, window peak %.1f MB (booked %.1f)\n",
                MetalMusic3Dit::i8_tier_name(eff), (double)held / 1048576.0,
                (double)booked / 1048576.0, worst, at0, bar0, short_d, deq,
                song, per_eval, ms, (double)used / 1048576.0,
                (double)wbook / 1048576.0);
    std::printf("[music3]   (loaded in %.1f s; the reference's bf16 song: "
                "%.4f; the short window %.0f ms an eval, %lld of its GEMMs "
                "int8; 8 latents %.4f from f16)\n", load_s, bar_s, short_ms,
                short_i8, tiny_d);
    EXPECT_TRUE(used <= wbook + (2u << 20));
    if (tier < 0) {
      EXPECT_TRUE(per_eval == 0 && deq == 0);
    } else {
      // Every projection of every block, at a full window.
      EXPECT_TRUE(per_eval == 4LL * base.layers);
      // Under 1024 rows accurate's f16 weight runs the f16 route -- the
      // f16 run's output, to the bit -- and a held layout the int8 route
      // (dequantized only under a floor the environment set).
      const long long all = 4LL * base.layers;
      if (eff == 0) {
        EXPECT_TRUE(deq == 0 && short_i8 == 0 && short_d == 0.0 &&
                    tiny_d == 0.0);
      } else {
        EXPECT_TRUE(env_floor ? deq == all : deq == 0 && short_i8 == all);
        EXPECT_TRUE(short_d > 0.0 && short_d < (eff <= 2 ? 0.02 : 0.08));
        EXPECT_TRUE(tiny_d > 0.0 && tiny_d < (eff <= 2 ? 0.02 : 0.08));
      }
      if (eff <= 2) {
        // The rotated tiers within the reference's own bf16 distance, at
        // step 0 (the state both references share) and over the song.
        EXPECT_TRUE(at0 < bar0 && song < bar_s);
      } else {
        // token_plain, unrotated, is further: an outlier channel sets its
        // token's scale over all of K. A gross bar only.
        EXPECT_TRUE(at0 < 2.0 * bar0 && song < 0.15);
      }
    }
    dit.reset();
  }
}

// ---- the AR stage ---------------------------------------------------------

namespace {

// The window index of golden c0 column j: column 0 is <|audio_end|>,
// column 1 + c code c.
int
c0_col_to_window_(int j)
{
  return j == 0 ? 0 : music3::kSemanticBase + (j - 1);
}

}  // namespace

TEST(minimax_music3, ar_matches_reference)
{
  const std::string mp = env_("VPIPE_MUSIC3_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_MUSIC3_GOLDEN");
  if (mp.empty() || gp.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const fs::path g = fs::path(gp) / "ar";
  const fs::path gb = fs::path(gp) / "ar_bf16";
  std::vector<int> cs, hs;
  const auto codes = golden_<std::int32_t>(g, "codes", &cs);    // [S, 8]
  const auto c0 = golden_<float>(g, "c0_logits");                // [S,2,16385]
  const auto c0b = golden_<float>(gb, "c0_logits");
  const auto dl = golden_<float>(g, "depth_logits");             // [S,7,2,1024]
  const auto dlb = golden_<float>(gb, "depth_logits");
  const auto hid = golden_<float>(g, "hiddens", &hs);            // [F, 32768]
  const auto hidb = golden_<float>(gb, "hiddens");
  const auto tids = golden_<std::int32_t>(g, "text_ids");        // [2, n]
  ASSERT_TRUE(cs.size() == 2 && hs.size() == 2 && !c0.empty());
  if (cs.size() != 2 || hs.size() != 2) { return; }
  const int S = cs[0];
  const int F = hs[0];
  std::vector<std::array<int, 8>> forced((std::size_t)S);
  for (int s = 0; s < S; ++s) {
    for (int i = 0; i < 8; ++i) {
      forced[(std::size_t)s][(std::size_t)i] = codes[(std::size_t)s * 8 + i];
    }
  }
  // The prompt the golden fed, through our cleaners and tokenizer.
  const FlexData meta = read_json_(g / "meta.json");
  const fs::path tj = fs::path(mp) / "tokenizer" / "tokenizer.json";
  auto tok = Tokenizer::from_huggingface_json(tj.string(), nullptr);
  music3::AddedTokens added;
  ASSERT_TRUE(tok != nullptr &&
              music3::read_added_tokens(tj.string(), &added, nullptr));
  const std::vector<std::int32_t> ids = music3::encode_text(
      *tok, added,
      music3::prompt_text(meta.as_object().at("caption").as_string(),
                          meta.as_object().at("lyrics").as_string()));
  const std::size_t n = tids.size() / 2;
  EXPECT_TRUE(ids == std::vector<std::int32_t>(tids.begin(),
                                               tids.begin() +
                                                   (std::ptrdiff_t)n));
  EXPECT_TRUE(music3::unconditional_ids(ids) ==
              std::vector<std::int32_t>(tids.begin() + (std::ptrdiff_t)n,
                                        tids.end()));

  // The reference's own bf16 distances, the bar.
  const double bar_c0 = rel_l2_(c0b.data(), c0.data(), c0.size());
  const double bar_dl = rel_l2_(dlb.data(), dl.data(), dl.size());
  const double bar_h = rel_l2_(hidb.data(), hid.data(), hid.size());
  std::printf("[music3] ref bf16 from fp32: c0 %.4f, depth %.4f, hiddens "
              "%.4f\n", bar_c0, bar_dl, bar_h);

  for (int quant : {8, 0}) {
    MetalMusic3Ar::Config cfg;
    std::string err;
    ASSERT_TRUE(MetalMusic3Ar::config_from_dir(mp, &cfg, &err));
    cfg.lm_quant_bits = quant;
    cfg.depth_quant_bits = quant;
    const auto t0 = std::chrono::steady_clock::now();
    auto ar = MetalMusic3Ar::load(
        WeightSet::open((fs::path(mp) / "language_model").string(), nullptr),
        WeightSet::open((fs::path(mp) / "rvq_depth_decoder").string(),
                        nullptr),
        mc, cfg, &err);
    if (ar == nullptr) { std::printf("[music3] ar: %s\n", err.c_str()); }
    ASSERT_TRUE(ar != nullptr);
    if (ar == nullptr) { return; }
    std::printf("[music3] AR %s loaded in %.1f s, %.2f GB (LM %s, depth "
                "%s)\n", quant == 8 ? "w8" : "bf16",
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count(),
                (double)ar->resident_bytes() / 1e9,
                ar->lm_quantized() ? "w8" : "bf16",
                ar->depth_quantized() ? "w8" : "bf16");
    EXPECT_TRUE(ar->lm_quantized() == (quant == 8));
    MetalMusic3Ar::Song song;
    MetalMusic3Ar::Forced rec;
    ASSERT_TRUE(ar->generate_forced(ids, F, forced, &song, &rec, &err));
    if (!err.empty()) { std::printf("[music3] %s\n", err.c_str()); }
    ASSERT_TRUE(song.frames == F && !song.ended);
    // c0: the golden's columns from our window rows.
    const int W = music3::kHeadRows;
    std::vector<float> ours(c0.size());
    for (int s = 0; s < S; ++s) {
      for (int b = 0; b < 2; ++b) {
        const float* src = rec.c0_logits.data() + ((std::size_t)s * 2 + b) * W;
        float* dst = ours.data() + ((std::size_t)s * 2 + b) * 16385;
        for (int j = 0; j < 16385; ++j) { dst[j] = src[c0_col_to_window_(j)]; }
      }
    }
    const double d_c0 = rel_l2_(ours.data(), c0.data(), c0.size());
    const double d_dl = rel_l2_(rec.depth_logits.data(), dl.data(), dl.size());
    std::vector<float> h32(song.hiddens.size());
    for (std::size_t i = 0; i < h32.size(); ++i) {
      h32[i] = (float)song.hiddens[i];
    }
    ASSERT_TRUE(h32.size() == hid.size());
    const double d_h = rel_l2_(h32.data(), hid.data(), hid.size());
    std::printf("[music3] AR %s from fp32: c0 %.4f, depth %.4f, hiddens "
                "%.4f (%d frames, %.1f s: %.1f ms a frame)\n",
                quant == 8 ? "w8" : "bf16", d_c0, d_dl, d_h, F, song.seconds,
                1e3 * (song.seconds - song.prefill_seconds) / S);
    // Per hidden slot (0 the LM's, 1..7 the depth decoder's).
    for (int slot = 0; slot < 8; ++slot) {
      double num = 0, den = 0;
      for (int f = 0; f < F; ++f) {
        for (int c = 0; c < 4096; ++c) {
          const std::size_t i = (std::size_t)f * 32768 + slot * 4096 + c;
          const double d = (double)h32[i] - hid[i];
          num += d * d;
          den += (double)hid[i] * hid[i];
        }
      }
      std::printf("[music3]   slot %d %.4f\n", slot, std::sqrt(num / den));
    }
    // The decisive c0 picks: the guided top-1 where fp32's margin over the
    // runner-up is at least 0.5.
    int decisive = 0, agree = 0;
    std::vector<float> sw, sg;
    for (int s = 0; s < S; ++s) {
      if (codes[(std::size_t)s * 8] < 0) { continue; }
      std::vector<float> fc(W), fu(W);
      for (int b = 0; b < 2; ++b) {
        std::vector<float>& dst = b == 0 ? fc : fu;
        std::fill(dst.begin(), dst.end(), -1e30f);
        const float* src = c0.data() + ((std::size_t)s * 2 + b) * 16385;
        for (int j = 0; j < 16385; ++j) {
          dst[(std::size_t)c0_col_to_window_(j)] = src[j];
        }
      }
      music3::FramePicker::semantic_scores(fc.data(), fu.data(), &sg);
      music3::FramePicker::semantic_scores(
          rec.c0_logits.data() + (std::size_t)s * 2 * W,
          rec.c0_logits.data() + ((std::size_t)s * 2 + 1) * W, &sw);
      std::vector<int> o(sg.size());
      for (std::size_t i = 0; i < o.size(); ++i) { o[i] = (int)i; }
      std::partial_sort(o.begin(), o.begin() + 2, o.end(), [&](int a, int b) {
        return sg[(std::size_t)a] > sg[(std::size_t)b];
      });
      if (sg[(std::size_t)o[0]] - sg[(std::size_t)o[1]] < 0.5f) { continue; }
      ++decisive;
      const int ours_top = (int)(std::max_element(sw.begin(), sw.end()) -
                                 sw.begin());
      agree += ours_top == o[0];
    }
    std::printf("[music3]   decisive c0 picks %d / %d\n", agree, decisive);
    EXPECT_TRUE(decisive > 0 && agree == decisive);
    // bf16 as close as the reference's own bf16 is; w8 (both stacks'
    // projections at affine 8-bit, group 64) MEASURED 1.22 / 1.18 / 1.29x
    // it on c0 / depth / hiddens, every decisive pick the same.
    const double lim = quant == 8 ? 1.35 : 1.0;
    EXPECT_TRUE(d_c0 < lim * bar_c0);
    EXPECT_TRUE(d_dl < lim * bar_dl);
    EXPECT_TRUE(d_h < lim * bar_h);
  }
}

// ---- the stages, end to end -----------------------------------------------

namespace {

class Music3PcmSink : public TypedStage<Music3PcmSink> {
public:
  static constexpr const char* kTypeName = "ut-music3-pcm-sink";
  using TypedStage::TypedStage;
  int beats = 0, channels = 0, rate = 0;
  std::int64_t samples = 0;
  double peak = 0.0, rms = 0.0;
  std::vector<float> data;
  Job process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    const auto* t = dynamic_cast<const TensorBeatPayload*>(p.get());
    if (t == nullptr || t->dtype != TensorBeat::DType::F32 ||
        t->shape.size() != 2) {
      co_return;
    }
    ++beats;
    channels = (int)t->shape[0];
    samples = t->shape[1];
    const float* d = t->as_f32();
    data.assign(d, d + t->element_count());
    double sq = 0.0;
    for (std::int64_t i = 0; i < t->element_count(); ++i) {
      peak = std::max(peak, (double)std::fabs(d[i]));
      sq += (double)d[i] * d[i];
    }
    rms = std::sqrt(sq / (double)std::max<std::int64_t>(1,
                                                       t->element_count()));
    if (t->sideband.is_object()) {
      auto sb = t->sideband.as_object();
      if (sb.contains("sample_rate")) {
        rate = (int)sb.at("sample_rate").as_int(0);
      }
    }
    co_return;
  }
};

class Music3Mono : public TypedStage<Music3Mono> {
public:
  static constexpr const char* kTypeName = "ut-music3-mono";
  using TypedStage::TypedStage;
  Job process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    const auto* t = dynamic_cast<const TensorBeatPayload*>(p.get());
    if (t == nullptr || t->dtype != TensorBeat::DType::F32 ||
        t->shape.size() != 2) {
      co_return;
    }
    const std::int64_t C = t->shape[0], N = t->shape[1];
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {1, N};
    out->resize_contiguous((std::size_t)N);
    const float* src = t->as_f32();
    float* dst = out->as_f32();
    for (std::int64_t i = 0; i < N; ++i) {
      float acc = 0.0f;
      for (std::int64_t c = 0; c < C; ++c) { acc += src[c * N + i]; }
      dst[i] = acc / (float)C;
    }
    out->sideband = t->sideband;
    co_await ctx.write(0, std::move(out));
  }
};

class Music3Transcript : public TypedStage<Music3Transcript> {
public:
  static constexpr const char* kTypeName = "ut-music3-transcript";
  using TypedStage::TypedStage;
  std::string text;
  Job process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    if (const auto* f = dynamic_cast<const FlexDataPayload*>(p.get())) {
      if (f->data.is_object() && f->data.as_object().contains("text")) {
        text += std::string(f->data.as_object().at("text").as_string(""));
        text += " ";
      }
    }
  }
};

// Lower-case words; section tags ("[Verse]") and Qwen3-ASR's leading
// "language X<asr_text>" dropped.
std::vector<std::string>
words_(std::string s)
{
  std::vector<std::string> out;
  std::string w;
  bool in_tag = false;
  std::size_t i = 0;
  while ((i = s.find("<asr_text>")) != std::string::npos) {
    std::size_t lang = s.rfind("language", i);
    if (lang == std::string::npos || lang > i) { lang = i; }
    s.erase(lang, i + 10 - lang);
  }
  for (unsigned char ch : s) {
    if (ch == '[') { in_tag = true; continue; }
    if (ch == ']') { in_tag = false; continue; }
    if (in_tag) { continue; }
    if (std::isalnum(ch) || ch == '\'') {
      w.push_back((char)std::tolower(ch));
    } else if (!w.empty()) {
      out.push_back(w);
      w.clear();
    }
  }
  if (!w.empty()) { out.push_back(w); }
  return out;
}

const char* kE2eCaption =
    "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, "
    "building gently into the chorus. Vocals: soft female lead, close and "
    "breathy. Arrangement: fingerpicked guitar and soft piano.";

}  // namespace

namespace {

struct E2eRun {
  bool launched = false, music3 = false, swapped = false;
  std::uint64_t songs = 0;
  int beats = 0, channels = 0, rate = 0;
  std::int64_t samples = 0;
  double peak = 0.0, rms = 0.0;
  std::vector<float> pcm;
  std::size_t device_peak = 0;     // above the run's start
  std::size_t footprint_peak = 0;  // the process's, whole
  std::size_t held = 0, booked = 0;
};

std::size_t
footprint_()
{
  task_vm_info_data_t ti;
  mach_msg_type_number_t c = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&ti, &c) !=
      KERN_SUCCESS) {
    return 0;
  }
  return (std::size_t)ti.phys_footprint;
}

// The 10 s graph -- generate-audio, audio-vae-decode, a sink -- once.
E2eRun
run_e2e_(const std::string& mp, const std::function<void(GenerateAudioStage*)>&
                                    check_plan = {})
{
  E2eRun r;
  Session sess;
  Pipeline pl("music3-e2e", &sess);
  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(mp));
    o.insert("style", FlexData::make_string(kE2eCaption));
    o.insert("lyrics", FlexData::make_string(
                           "[verse]\nMorning light filtering through the pine\n"
                           "[chorus]\nSoftly the world begins to breathe"));
    o.insert("max_seconds", FlexData::make_real(10.0));
    o.insert("seed", FlexData::make_int(7));
  }
  auto* gen = static_cast<GenerateAudioStage*>(pl.insert_stage(
      std::make_unique<GenerateAudioStage>(&sess, "gen", std::vector<InEdge>{},
                                           std::move(cfg))));
  r.music3 = gen->family() == GenerateAudioStage::Family::kMusic3;
  r.swapped = gen->music3_swaps();
  if (check_plan) { check_plan(gen); }
  FlexData vcfg = FlexData::make_object();
  vcfg.as_object().insert("hf_dir", FlexData::make_string(mp));
  auto* dec = static_cast<AudioVaeDecodeStage*>(pl.insert_stage(
      std::make_unique<AudioVaeDecodeStage>(
          &sess, "dec", std::vector<InEdge>{{gen, 0}}, std::move(vcfg))));
  auto* pcm = static_cast<Music3PcmSink*>(pl.insert_stage(
      std::make_unique<Music3PcmSink>(&sess, "pcm",
                                      std::vector<InEdge>{{dec, 0}},
                                      FlexData::make_object())));
  MetalCompute* mc = sess.metal_compute();
  const std::size_t alloc0 = mc != nullptr ? mc->memory_budget().allocated
                                           : 0;
  std::atomic<std::size_t> peak{alloc0};
  std::atomic<std::size_t> fpeak{footprint_()};
  std::atomic<bool> done{false};
  std::thread sampler([&] {
    while (mc != nullptr && !done.load()) {
      const std::size_t a = mc->memory_budget().allocated;
      if (a > peak.load()) { peak.store(a); }
      const std::size_t f = footprint_();
      if (f > fpeak.load()) { fpeak.store(f); }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
  PipelineRuntime rt(&pl, &sess);
  const bool launched = rt.launch();
  if (launched) { rt.wait_idle(); }
  // What the song left held beside the weights, before the stop.
  r.held = gen->held_scratch_bytes();
  r.booked = gen->planned_song_bytes().held +
             gen->planned_song_bytes().flow;
  rt.stop();
  done.store(true);
  sampler.join();
  r.launched = launched;
  r.device_peak = peak.load() - alloc0;
  r.footprint_peak = fpeak.load();
  r.songs = gen->songs_emitted();
  std::printf("[music3] e2e: VAE family '%s', %d beat(s), %d ch @ %d Hz, "
              "%lld samples (%.2f s), peak %.3f rms %.4f\n",
              dec->family().c_str(), pcm->beats, pcm->channels, pcm->rate,
              (long long)pcm->samples,
              pcm->rate > 0 ? (double)pcm->samples / pcm->rate : 0.0,
              pcm->peak, pcm->rms);
  r.beats = pcm->beats;
  r.channels = pcm->channels;
  r.rate = pcm->rate;
  r.samples = pcm->samples;
  r.peak = pcm->peak;
  r.rms = pcm->rms;
  r.pcm = pcm->data;
  return r;
}

}  // namespace

// The stage end to end, twice: on this box as it is, and with the halves
// TAKING TURNS (the language model let go before the flow matching loads)
// as they do on a 16 GB box, here by VPIPE_RAM_LIMIT_MB 16384 -- the plan
// phases the halves' claims, the device's peak falls by about the half
// not held, and the song is the same to the bit.
TEST(minimax_music3, generate_audio_end_to_end)
{
  const std::string mp = env_("VPIPE_MUSIC3_TEST_MODEL_PATH");
  if (mp.empty()) { return; }
  unsetenv("VPIPE_MUSIC3_SWAP");
  auto check = [&](const E2eRun& r) {
    EXPECT_TRUE(r.music3 && r.launched && r.songs == 1);
    EXPECT_TRUE(r.beats == 1 && r.channels == 2 && r.rate == 44100);
    // At most 250 frames (the 10 s cap), stitched to whole latents.
    EXPECT_TRUE(r.samples > 0 &&
                r.samples <= (std::int64_t)512 *
                                 music3::stitched_latents(250) &&
                r.samples % 512 == 0);
    EXPECT_TRUE(r.peak > 0.01 && r.peak <= 1.0 && r.rms > 1e-3);
  };
  const E2eRun both = run_e2e_(mp);
  check(both);
  std::printf("[music3] e2e: halves %s, device peak %.1f MB, process "
              "footprint peak %.1f MB, held beside the weights %zu MB, "
              "booked %zu MB\n", both.swapped ? "in turn" : "both held",
              (double)both.device_peak / 1048576.0,
              (double)both.footprint_peak / 1048576.0, both.held >> 20,
              both.booked >> 20);
  // The K/V pool at its high-water capacity and the DiT's window, inside
  // what the plan booked for a song at the config's bounds.
  EXPECT_TRUE(both.held > 0 && both.held <= both.booked);

  setenv("VPIPE_RAM_LIMIT_MB", "16384", 1);
  const E2eRun turns = run_e2e_(mp, [&](GenerateAudioStage* gen) {
    // The plan: the AR half's weights and run in the condition phase, the
    // flow half's in denoise, the hidden states from one to the other.
    int cond = 0, den = 0, handoff = 0;
    for (const ResourceClaim& c : gen->declare_resources()) {
      const bool ar = c.key.find("language_model") != std::string::npos ||
                      c.key.find("rvq_depth_decoder") != std::string::npos ||
                      c.key.find("/held") != std::string::npos;
      const bool fl = c.key.find("transformer") != std::string::npos ||
                      c.key.find("condition_encoder") != std::string::npos ||
                      c.key.find("/flow") != std::string::npos;
      if (ar && c.phase == model_memory::kPhaseCondition) { ++cond; }
      if (fl && c.phase == model_memory::kPhaseDenoise) { ++den; }
      if (c.key.find("/handoff") != std::string::npos &&
          c.phase == model_memory::kPhaseCondition &&
          c.last_phase == model_memory::kPhaseDenoise) {
        ++handoff;
      }
    }
    std::printf("[music3] e2e at 16 GB: %d condition, %d denoise and %d "
                "hand-off claims\n", cond, den, handoff);
    EXPECT_TRUE(cond == 3 && den == 3 && handoff == 1);
  });
  unsetenv("VPIPE_RAM_LIMIT_MB");
  check(turns);
  std::printf("[music3] e2e at 16 GB: halves %s, device peak %.1f MB, "
              "process footprint peak %.1f MB\n",
              turns.swapped ? "in turn" : "both held",
              (double)turns.device_peak / 1048576.0,
              (double)turns.footprint_peak / 1048576.0);
  EXPECT_TRUE(turns.swapped && !both.swapped);
  // The same song, sample for sample.
  EXPECT_TRUE(!both.pcm.empty() && both.pcm == turns.pcm);
  // The peak is one half's: at least the smaller half (the flow matching,
  // 4.7 GB) below holding both.
  EXPECT_TRUE(turns.device_peak + (4ull << 30) < both.device_peak);
  // ... and so does the whole process's: no host copy of a half lingers
  // (a loader's freed host vectors would stay in it, dirty).
  EXPECT_TRUE(turns.footprint_peak + (4ull << 30) < both.footprint_peak);
}

// SUNG LYRICS, HEARD BACK: a 45 s song through the stages, a downmix, a
// 16 kHz resample and Qwen3-ASR; held is the share of the lyrics' words
// heard back. The request matters: with a one-line description and a
// four-line lyric the model opens with wordless singing for its first
// ~30 s -- so does the reference's own render of the golden's codes --
// so this one is a structured caption and the official example's first
// verse and pre-chorus (MiniMax's demo of them, through the same chain,
// is heard verbatim). Opt-in (VPIPE_MUSIC3_ASR_CHECK=1, plus
// VPIPE_QWEN3_ASR_TEST_MODEL_PATH): a 45 s song is minutes.
TEST(minimax_music3, sung_lyrics_transcribe)
{
  const std::string mp = env_("VPIPE_MUSIC3_TEST_MODEL_PATH");
  const std::string ap = env_("VPIPE_QWEN3_ASR_TEST_MODEL_PATH");
  if (mp.empty() || ap.empty() || env_("VPIPE_MUSIC3_ASR_CHECK").empty()) {
    return;
  }
  const std::string caption =
      "Global Metadata\n"
      "Basic Attributes: bpm is 92. key is E, and scale is minor. Electric "
      "Blues / Blues Rock.\n"
      "Global Emotional Progression: A confident, gritty swagger from the "
      "first bar, a driving mid-tempo groove throughout.\n"
      "Vocal Details\n"
      "Vocal Gender & Timbre: Singer A (Male). A deep, gravelly baritone.\n"
      "Vocal Style: Conversational and storytelling-oriented in the "
      "verses, melodic and soulful in the refrains; the lyrics stay clear.\n"
      "Arrangement\n"
      "Primary: A slightly overdriven electric guitar from intro to outro.\n"
      "Secondary: A walking electric bass and a Hammond-style organ.\n"
      "Groove & Foundation Progression: A steady shuffle on an acoustic "
      "drum kit.";
  const std::string lyrics =
      "[verse]\nI'm learning how to fill up\nevery space I used to leave,\n"
      "teaching my own heart the patience\nthat my family needs from me.\n"
      "And I'm learning how to soften,\nhow to rise above the past\n"
      "[pre-chorus]\nBreathe a little deeper,\nlove is here to heal.\n"
      "Let love lift what used to fall\n";
  const std::vector<std::string> said = words_(lyrics);
  const std::string q = env_("VPIPE_MUSIC3_ASR_QUANT").empty()
                            ? "w8" : env_("VPIPE_MUSIC3_ASR_QUANT");
  Session sess;
  Pipeline pl("music3-asr", &sess);
  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(mp));
    o.insert("style", FlexData::make_string(caption));
    o.insert("lyrics", FlexData::make_string(lyrics));
    o.insert("max_seconds", FlexData::make_real(45.0));
    o.insert("seed", FlexData::make_int(20261028));
    o.insert("lm_quant", FlexData::make_string(q));
  }
  auto* gen = static_cast<GenerateAudioStage*>(pl.insert_stage(
      std::make_unique<GenerateAudioStage>(&sess, "gen",
                                           std::vector<InEdge>{},
                                           std::move(cfg))));
  FlexData vcfg = FlexData::make_object();
  vcfg.as_object().insert("hf_dir", FlexData::make_string(mp));
  auto* dec = pl.insert_stage(std::make_unique<AudioVaeDecodeStage>(
      &sess, "dec", std::vector<InEdge>{{gen, 0}}, std::move(vcfg)));
  auto mono_st = std::make_unique<Music3Mono>(
      &sess, "mono", std::vector<InEdge>{{dec, 0}}, FlexData::make_object());
  mono_st->allocate_oports(1);
  auto* mono = pl.insert_stage(std::move(mono_st));
  FlexData rcfg = FlexData::make_object();
  rcfg.as_object().insert("output_sample_rate", FlexData::make_int(16000));
  auto* rs = pl.insert_stage(std::make_unique<AudioTemporalResampleStage>(
      &sess, "resample", std::vector<InEdge>{{mono, 0}}, std::move(rcfg)));
  FlexData acfg = FlexData::make_object();
  {
    auto o = acfg.as_object();
    o.insert("hf_dir", FlexData::make_string(ap));
    o.insert("max_new_tokens", FlexData::make_int(256));
    o.insert("pcm_buffer_s", FlexData::make_real(60.0));
  }
  auto* asr = pl.insert_stage(std::make_unique<AudioTranscribeStage>(
      &sess, "asr", std::vector<InEdge>{{rs, 0}}, std::move(acfg)));
  auto* sink = static_cast<Music3Transcript*>(pl.insert_stage(
      std::make_unique<Music3Transcript>(&sess, "sink",
                                         std::vector<InEdge>{{asr, 0}},
                                         FlexData::make_object())));
  PipelineRuntime rt(&pl, &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  EXPECT_TRUE(gen->songs_emitted() == 1);
  std::vector<std::string> heard = words_(sink->text);
  int hit = 0;
  for (const std::string& w : said) {
    auto it = std::find(heard.begin(), heard.end(), w);
    if (it != heard.end()) { ++hit; heard.erase(it); }
  }
  const double recall = said.empty() ? 0.0 : (double)hit / said.size();
  std::printf("[music3] lm_quant %s: %d/%zu lyric words heard (%.2f)\n"
              "  heard '%s'\n", q.c_str(), hit, said.size(), recall,
              sink->text.c_str());
  EXPECT_TRUE(recall >= 0.5);
}

// The repository is detected as what it is: model-register and the
// pickers (generate-audio's and audio-vae-decode's hf_dir) filter on it.
TEST(minimax_music3, detects_the_repository)
{
  const std::string mp = env_("VPIPE_MUSIC3_TEST_MODEL_PATH");
  if (mp.empty()) { return; }
  const DetectedModel d = detect_model_dir(mp, "MiniMaxAI/MiniMax-Music3");
  std::printf("[music3] detected '%s' by %s\n", d.model_type.c_str(),
              d.detected_by.c_str());
  EXPECT_TRUE(d.model_type == "minimax-music3");
  // Without the catalogue's name: by its config.json.
  const DetectedModel c = detect_model_dir(mp, "");
  std::printf("[music3] unnamed: '%s' by %s\n", c.model_type.c_str(),
              c.detected_by.c_str());
  EXPECT_TRUE(c.model_type == "minimax-music3");
}

// THE PLAN AGAINST THE DEVICE. Each part is booked at what it holds
// (generate-audio's claims carry kClaimHeldBytes; the planner never books
// a part above its checkpoint), and a song's working set at
// MetalMusic3Ar::run_bytes + MetalMusic3Dit::window_scratch_bytes. Here:
// every part's estimate against what it holds once loaded, then the peak
// device allocation (sampled from a thread) over a song from a ~2400-
// token prompt -- its prefill pieces, both contexts' K/V, 250 frames, two
// flow-matching windows -- against the run figure for that prompt and
// length. The tuners are pinned: their probes are transient buffers of
// their own.
TEST(minimax_music3, the_plan_books_what_a_song_holds)
{
  const std::string mp = env_("VPIPE_MUSIC3_TEST_MODEL_PATH");
  if (mp.empty()) { return; }
  setenv("VPIPE_QWEN_GQA_AUTOTUNE", "0", 1);
  setenv("VPIPE_QWEN_PREFILL_AUTOTUNE", "0", 1);
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const fs::path root(mp);
  MetalMusic3Ar::Config ac;
  MetalMusic3Dit::Config dc;
  std::string err;
  ASSERT_TRUE(MetalMusic3Ar::config_from_dir(mp, &ac, &err) &&
              MetalMusic3Dit::config_from_dir(mp, &dc, &err));
  auto ar = MetalMusic3Ar::load(
      WeightSet::open((root / "language_model").string(), nullptr),
      WeightSet::open((root / "rvq_depth_decoder").string(), nullptr), mc,
      ac, &err);
  auto dit = MetalMusic3Dit::load(
      WeightSet::open((root / "transformer").string(), nullptr),
      WeightSet::open((root / "condition_encoder").string(), nullptr), mc,
      dc, &err);
  ASSERT_TRUE(ar != nullptr && dit != nullptr);
  if (ar == nullptr || dit == nullptr) { return; }

  // 1. The weights: estimate == held, per part.
  struct Part { const char* name; std::size_t est, held; };
  const Part parts[] = {
      {"language_model", MetalMusic3Ar::lm_weight_bytes(mp, 8),
       ar->resident_bytes() - ar->depth_bytes()},
      {"rvq_depth_decoder", MetalMusic3Ar::depth_weight_bytes(mp, 8),
       ar->depth_bytes()},
      {"condition_encoder", MetalMusic3Dit::condition_weight_bytes(mp),
       dit->condition_bytes()},
      {"transformer", MetalMusic3Dit::transformer_weight_bytes(mp),
       dit->resident_bytes() - dit->condition_bytes()}};
  for (const Part& p : parts) {
    const double r = (double)p.est / (double)std::max<std::size_t>(p.held, 1);
    std::printf("[music3] %-18s booked %8.1f MB, holds %8.1f MB (%.4f)\n",
                p.name, (double)p.est / 1048576.0,
                (double)p.held / 1048576.0, r);
    // The backbone's own buffers (norms, its small scratch) are not in
    // the estimate's tensor walk to the byte; everything else is exact.
    EXPECT_TRUE(r > 0.995 && r < 1.005);
  }

  // 2. A song's working set against the device.
  const fs::path tj = root / "tokenizer" / "tokenizer.json";
  auto tok = Tokenizer::from_huggingface_json(tj.string(), nullptr);
  music3::AddedTokens added;
  ASSERT_TRUE(tok != nullptr &&
              music3::read_added_tokens(tj.string(), &added, nullptr));
  std::string caption;
  for (int i = 0; i < 40; ++i) { caption += std::string(kE2eCaption) + "\n"; }
  const auto ids = music3::encode_text(
      *tok, added,
      music3::prompt_text(caption, "[verse]\nMorning light filtering "
                                   "through the pine\n[chorus]\nSoftly the "
                                   "world begins to breathe"));
  const int frames = 250;
  const std::size_t booked =
      MetalMusic3Ar::run_bytes(ac, (int)ids.size(), frames,
                               mc->supports_matrix_cores()) +
      MetalMusic3Ar::handoff_bytes(ac, frames) +
      MetalMusic3Dit::window_scratch_bytes(dc);
  const std::size_t alloc0 = mc->memory_budget().allocated;
  std::atomic<std::size_t> peak{alloc0};
  std::atomic<bool> done{false};
  std::thread sampler([&] {
    while (!done.load()) {
      const std::size_t a = mc->memory_budget().allocated;
      if (a > peak.load()) { peak.store(a); }
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  });
  MetalMusic3Ar::Song song;
  const bool ok = ar->generate(ids, frames, 7, {}, &song, &err);
  std::vector<float> lat;
  const bool ok2 =
      ok && dit->synthesize(song.hiddens.data(), song.frames, 7, 2, 1.7f,
                            &lat, {}, &err);
  done.store(true);
  sampler.join();
  ASSERT_TRUE(ok && ok2);
  const std::size_t used =
      std::max(peak.load(), mc->memory_budget().allocated) - alloc0;
  // The handoff is host memory the device does not count; the run figure
  // books it, so the device's peak is held to the figure without it.
  const std::size_t handoff = MetalMusic3Ar::handoff_bytes(ac, song.frames);
  std::printf("[music3] a %zu-token prompt + %d frames + 2 windows peaked "
              "%.1f MB on the device; booked %.1f MB (%.1f without the "
              "%.1f MB host handoff)\n", ids.size(), song.frames,
              (double)used / 1048576.0, (double)booked / 1048576.0,
              (double)(booked - handoff) / 1048576.0,
              (double)handoff / 1048576.0);
  // The K/V pool as it stands, beside what a context's tokens fill: the
  // figure books the pool at the capacity its doubling CAN reach (up to
  // twice the pages), which is most of the margin above.
  const std::size_t kv_need =
      2 * ((std::size_t)ids.size() + (std::size_t)frames + 1) *
      (std::size_t)ac.layers * 2 * ac.kv_heads * ac.head_dim * 2;
  std::printf("[music3] the K/V pool holds %.1f MB for %.1f MB of tokens\n",
              (double)ar->held_scratch_bytes() / 1048576.0,
              (double)kv_need / 1048576.0);
  // The flow matching's window scratch, allocated at the first 200-frame
  // window and kept, against its booking.
  std::printf("[music3] the DiT's window scratch %.1f MB, booked %.1f MB\n",
              (double)dit->scratch_bytes() / 1048576.0,
              (double)MetalMusic3Dit::window_scratch_bytes(dc) / 1048576.0);
  EXPECT_TRUE(dit->scratch_bytes() <=
              MetalMusic3Dit::window_scratch_bytes(dc));
  // The plan at the protocol's cap: a 5000-token prompt, 9000 frames.
  std::printf("[music3] at the cap the plan books the AR's run %.1f MB, the "
              "hand-off %.1f MB, the window %.1f MB\n",
              (double)MetalMusic3Ar::run_bytes(
                  ac, music3::kMaxPromptTokens, music3::kMaxFrames,
                  mc->supports_matrix_cores()) / 1048576.0,
              (double)MetalMusic3Ar::handoff_bytes(ac, music3::kMaxFrames) /
                  1048576.0,
              (double)MetalMusic3Dit::window_scratch_bytes(dc) / 1048576.0);
  EXPECT_TRUE(ids.size() > 1536 && song.frames == frames);
  EXPECT_TRUE(used <= booked - handoff + (2u << 20));
}
