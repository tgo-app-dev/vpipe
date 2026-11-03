// MiniMax-H3 GUIDES through the denoise loop on the DiT's own blocks.
//
// A guide (generate-video iport12 / iport13) puts conditioning AUDIO rows
// in a keyframe-layout request for the first time -- before it only a
// Ref2VA request led its audio buffer with rows the loop must not step.
// Two real blocks are enough to hold the loop to that: the guide's rows
// come out exactly as they went in, the generated rows move, and the
// guide changes what is generated. Whether a guide makes a continuation
// redo its tail is a question for the whole model, not two blocks.
//
// VPIPE_MINIMAX_H3_TEST_MODEL_PATH names the checkpoint (an H3 root or
// its FL2VA partition).

#include "minitest.h"
#include "minimax-h3-block-run.h"
#include "common/session.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-denoise.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;

namespace {

std::vector<float>
noise_(std::size_t n, std::uint64_t seed)
{
  std::vector<float> v(n);
  std::mt19937_64 g(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (float& x : v) { x = nd(g); }
  return v;
}

}  // namespace

TEST(minimax_h3_guide, the_loop_holds_a_guide_and_reads_it)
{
  const std::string root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  BlockRun br;
  ASSERT_TRUE(br.setup(mc, root, 2));
  if (br.vb.empty()) { return; }
  const auto& c = br.cfg;
  auto m = MetalMiniMaxH3Transformer::load(br.root, mc, c, false, {});
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  m->set_gemm_route(MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32);

  // 7 latent frames (22 pixel frames) of 12 x 20 and their soundtrack;
  // the guide is 2 latent frames from frame 0 and 8 audio latents.
  const int lt = 7, lh = 12, lw = 20;
  const int alat = h3b::audio_latent_num_frames(h3b::guide_pixel_frames(lt));
  const int PE = c.video_patch_elems(), AC = c.audio_channels;
  const int rpf = (lh / c.patch_h) * (lw / c.patch_w);
  const std::vector<int> tags(8, h3b::kTextTag);
  const std::vector<float> gvid =
      noise_((std::size_t)c.video_channels * 2 * lh * lw, 11);
  const std::vector<float> gaud = noise_((std::size_t)2 * AC * 8, 12);
  // The generated rows' noise is the same in every run, whatever leads
  // it, so a difference between runs is the guide's.
  const std::vector<float> tv = noise_((std::size_t)lt * rpf * PE, 13);
  const std::vector<float> ta = noise_((std::size_t)2 * alat * AC, 14);

  struct Run {
    h3b::PackedLayout L;
    std::vector<float> v, a, v0, a0;
    bool ok = false;
    std::string err;
  };
  auto run = [&](const std::vector<h3b::Guide>& guides) {
    Run r;
    if (!h3b::build_packed_sequence(tags, lt, lh, lw, alat, c.patch_h,
                                    c.patch_w, h3b::kAudioChannels, {},
                                    &r.L, guides)) {
      r.err = "layout";
      return r;
    }
    const int gv = r.L.num_condition_video_rows;
    const int ga = r.L.num_condition_audio_rows;
    r.v.resize(r.L.video_indices.size() * (std::size_t)PE);
    r.a.resize(r.L.audio_indices.size() * (std::size_t)AC);
    for (const h3b::Guide& g : guides) {
      if (g.num_latent_frames > 0) {
        h3b::patchify_latent_frames(gvid.data(), c.video_channels, 2, 0,
                                    g.num_latent_frames, lh, lw, c.patch_h,
                                    c.patch_w, r.v.data());
      }
      if (g.num_audio_latents > 0) {
        h3b::pack_audio_latent_rows(gaud.data(), AC, 8, g.num_audio_latents,
                                    r.a.data());
      }
    }
    std::memcpy(r.v.data() + (std::size_t)gv * PE, tv.data(),
                tv.size() * sizeof(float));
    std::memcpy(r.a.data() + (std::size_t)ga * AC, ta.data(),
                ta.size() * sizeof(float));
    r.v0 = r.v;
    r.a0 = r.a;
    DenoiseRequest q;
    q.dit = m.get();
    q.layout = &r.L;
    q.text = &br.tb;
    q.video = r.v.data();
    q.audio = r.a.data();
    q.video_grid_h = lh / c.patch_h;
    q.video_grid_w = lw / c.patch_w;
    q.sigma_grid = {1.0f, 0.6f, 0.0f};
    q.num_steps = 2;
    q.seed = 5;
    r.ok = denoise(q, &r.err);
    return r;
  };
  auto same = [](const std::vector<float>& a, const std::vector<float>& b,
                 std::size_t from, std::size_t n) {
    return std::memcmp(a.data() + from, b.data() + from,
                       n * sizeof(float)) == 0;
  };
  auto tail = [](const std::vector<float>& v, std::size_t from) {
    return std::vector<float>(v.begin() + (std::ptrdiff_t)from, v.end());
  };

  const Run none = run({});
  const Run both = run({{0, 2, 8}});
  const Run audio = run({{0, 0, 8}});
  ASSERT_TRUE(none.ok && both.ok && audio.ok);
  if (!none.ok || !both.ok || !audio.ok) {
    std::printf("  %s / %s / %s\n", none.err.c_str(), both.err.c_str(),
                audio.err.c_str());
    return;
  }
  const std::size_t gvv = (std::size_t)both.L.num_condition_video_rows * PE;
  const std::size_t gaa = (std::size_t)both.L.num_condition_audio_rows * AC;
  const std::size_t aaa = (std::size_t)audio.L.num_condition_audio_rows * AC;
  // The guide's rows, held: exactly what went in.
  const bool held = same(both.v, both.v0, 0, gvv) &&
                    same(both.a, both.a0, 0, gaa) &&
                    same(audio.a, audio.a0, 0, aaa);
  // The generated rows stepped, and moved by the guide.
  const double stepped = rel_l2_(tail(both.v, gvv), tv);
  const double dv = rel_l2_(tail(both.v, gvv), none.v);
  const double da = rel_l2_(tail(both.a, gaa), none.a);
  const double dva = rel_l2_(audio.v, none.v);
  const double daa = rel_l2_(tail(audio.a, aaa), none.a);
  std::printf("  guide rows held: %s (video %zu + audio %zu values); the "
              "generated video stepped %.3e from its noise; the guide "
              "moved it %.3e (video) / %.3e (audio); audio alone %.3e / "
              "%.3e; seq %d against %d\n", held ? "IDENTICAL" : "WRITTEN",
              gvv, gaa, stepped, dv, da, dva, daa, both.L.seq_len,
              none.L.seq_len);
  EXPECT_TRUE(both.L.num_condition_video_rows == 2 * rpf);
  EXPECT_TRUE(both.L.num_condition_audio_rows == 2 * 8);
  EXPECT_TRUE(held);
  EXPECT_TRUE(stepped > 1e-2);
  EXPECT_TRUE(dv > 1e-3 && da > 1e-3);
  EXPECT_TRUE(dva > 1e-3 && daa > 1e-3);
}
