// MiniMax-H3 `ref2va` reference encoding: the parts that decide WHAT the
// models are shown, checked without loading any of them.
//
// The reference encoder pairs three models with one list of media, and
// the pairing is where it goes wrong quietly -- the tower reads a clip
// at 2 fps on its own canvas while the VAE reads it at 24 on
// MiniMax-H3's, so feeding either the other's pixels still produces a
// well-shaped tensor. Two pieces of that decision are pure arithmetic
// and are pinned here: which frames the conditioner sees (and how their
// vision blocks are labelled), and what counts as a legal request.
//
// The frame goldens come from diffusers'
// `MiniMaxH3Ref2VATextEncoderStep._sample_video_condition_frames`.

#include "tests/minitest.h"

#ifdef VPIPE_BUILD_APPLE_SILICON

#include "generative-models/minimax-h3/minimax-h3-reference-encoder.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

namespace h3 = vpipe::genai::minimax_h3;

struct FrameCase {
  int                n;
  double             sample_fps;
  std::vector<int>   indices;
  std::vector<float> blocks;
};

// One entry per case: the source frame count at MiniMax-H3's own 24 fps,
// the rate the conditioner reads at, the frames it lands on, and the
// timestamp of every merged vision block.
//
// The three fractional strides are the point of the table. At 2 fps the
// stride is a whole 12 frames and any rounding rule agrees; at 5, 7 and
// 9 fps it is 4.8, 3.43 and 2.67, so the indices are NOT an arithmetic
// sequence and a rule that truncated where the reference rounds (or
// that skipped the deduplication) would diverge within a few entries.
const FrameCase kFrames[] = {
  {121, 2.0,
   {0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120},
   {0.250000f, 1.250000f, 2.250000f, 3.250000f, 4.250000f, 5.000000f}},
  {22, 2.0,
   {0, 12},
   {0.250000f}},
  {56, 2.0,
   {0, 12, 24, 36, 48},
   {0.250000f, 1.250000f, 2.000000f}},
  {39, 2.0,
   {0, 12, 24, 36},
   {0.250000f, 1.250000f}},
  {240, 2.0,
   {0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180,
    192, 204, 216, 228},
   {0.250000f, 1.250000f, 2.250000f, 3.250000f, 4.250000f, 5.250000f,
    6.250000f, 7.250000f, 8.250000f, 9.250000f}},
  {121, 5.0,
   {0, 5, 10, 14, 19, 24, 29, 34, 38, 43, 48, 53, 58, 62, 67, 72, 77, 82,
    86, 91, 96, 101, 106, 110, 115, 120},
   {0.100000f, 0.500000f, 0.900000f, 1.300000f, 1.700000f, 2.100000f,
    2.500000f, 2.900000f, 3.300000f, 3.700000f, 4.100000f, 4.500000f,
    4.900000f}},
  {56, 7.0,
   {0, 3, 7, 10, 14, 17, 21, 24, 27, 31, 34, 38, 41, 45, 48, 51, 55},
   {0.071429f, 0.357143f, 0.642857f, 0.928571f, 1.214286f, 1.500000f,
    1.785714f, 2.071429f, 2.285714f}},
  {30, 9.0,
   {0, 3, 5, 8, 11, 13, 16, 19, 21, 24, 27, 29},
   {0.055556f, 0.277778f, 0.500000f, 0.722222f, 0.944444f, 1.166667f}},
};

h3::MediaReference
image_ref_()
{
  h3::MediaReference m;
  m.kind = h3::MediaReference::Kind::kImage;
  m.num_frames = 1;
  m.height = 64;
  m.width = 64;
  m.rgb.assign(3 * 64 * 64, 128);
  return m;
}

h3::MediaReference
audio_ref_()
{
  h3::MediaReference m;
  m.kind = h3::MediaReference::Kind::kAudio;
  m.channels = 2;
  m.sample_rate = 32000;
  m.pcm.assign(2 * 32000, 0.0f);
  return m;
}

h3::MediaReference
video_ref_()
{
  h3::MediaReference m = image_ref_();
  m.kind = h3::MediaReference::Kind::kVideo;
  m.num_frames = 24;
  m.rgb.assign((std::size_t)24 * 3 * 64 * 64, 128);
  return m;
}

}  // namespace

TEST(minimax_h3_refenc, condition_frames_match_the_reference)
{
  int checked = 0;
  for (const FrameCase& c : kFrames) {
    std::vector<int> idx;
    std::vector<float> blocks;
    std::string err;
    ASSERT_TRUE(h3::condition_frame_indices(c.n, h3::kFps, c.sample_fps, 2,
                                            &idx, &blocks, &err));
    EXPECT_TRUE(idx.size() == c.indices.size());
    EXPECT_TRUE(blocks.size() == c.blocks.size());
    if (idx.size() != c.indices.size() ||
        blocks.size() != c.blocks.size()) {
      std::printf("[refenc] %d frames at %.1f fps: %zu/%zu indices, "
                  "%zu/%zu blocks\n", c.n, c.sample_fps, idx.size(),
                  c.indices.size(), blocks.size(), c.blocks.size());
      continue;
    }
    for (std::size_t i = 0; i < idx.size(); ++i) {
      const bool ok = idx[i] == c.indices[i];
      EXPECT_TRUE(ok);
      if (!ok) {
        std::printf("[refenc] %d@%.1f index %zu: %d, expected %d\n", c.n,
                    c.sample_fps, i, idx[i], c.indices[i]);
        break;
      }
    }
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      // The label is rendered "%.1f", so a block timestamp only has to
      // agree to well inside a tenth -- but these are exact rationals on
      // both sides, so the bar is float precision rather than the label.
      const bool ok = std::fabs(blocks[i] - c.blocks[i]) < 1e-5f;
      EXPECT_TRUE(ok);
      if (!ok) {
        std::printf("[refenc] %d@%.1f block %zu: %.6f, expected %.6f\n", c.n,
                    c.sample_fps, i, blocks[i], c.blocks[i]);
        break;
      }
    }
    ++checked;
  }
  std::printf("[minimax_h3_refenc] %d frame-sampling cases match the "
              "reference\n", checked);
  EXPECT_TRUE(checked == (int)(sizeof(kFrames) / sizeof(kFrames[0])));
}

TEST(minimax_h3_refenc, a_clip_too_short_to_merge_is_named)
{
  // Qwen3-VL merges the sampled frames in PAIRS, so one sampled frame is
  // not a vision block. Left to the tower it would be padded into one
  // silently, which conditions the model on a still it was told is
  // motion.
  std::vector<int> idx;
  std::vector<float> blocks;
  std::string err;
  EXPECT_FALSE(h3::condition_frame_indices(11, h3::kFps, 2.0, 2, &idx,
                                           &blocks, &err));
  EXPECT_TRUE(!err.empty());
  // 13 frames is the first length that samples two: round(1 * 12) + 1.
  EXPECT_TRUE(h3::condition_frame_indices(13, h3::kFps, 2.0, 2, &idx,
                                          &blocks, &err));
  EXPECT_TRUE(idx.size() == 2 && blocks.size() == 1);
  std::printf("[minimax_h3_refenc] a 11-frame clip is refused (\"%s\"), 13 "
              "frames sample %zu\n", err.c_str(), idx.size());
}

TEST(minimax_h3_refenc, request_limits)
{
  h3::ReferenceLimits lim;
  std::string err;

  // Empty is not a degenerate ref2va, it is a t2va request.
  EXPECT_FALSE(h3::validate_reference_request({}, lim, &err));

  // Audio alone never reaches the conditioner, so a request of nothing
  // but audio packs text rows describing references the model cannot
  // see.
  EXPECT_FALSE(h3::validate_reference_request({audio_ref_()}, lim, &err));
  EXPECT_FALSE(h3::validate_reference_request(
      {audio_ref_(), audio_ref_()}, lim, &err));
  EXPECT_TRUE(h3::validate_reference_request(
      {audio_ref_(), image_ref_()}, lim, &err));

  // The per-modality caps.
  std::vector<h3::MediaReference> many;
  for (int i = 0; i < lim.max_images; ++i) { many.push_back(image_ref_()); }
  EXPECT_TRUE(h3::validate_reference_request(many, lim, &err));
  many.push_back(image_ref_());
  EXPECT_FALSE(h3::validate_reference_request(many, lim, &err));

  std::vector<h3::MediaReference> vids;
  for (int i = 0; i < lim.max_videos; ++i) { vids.push_back(video_ref_()); }
  EXPECT_TRUE(h3::validate_reference_request(vids, lim, &err));
  vids.push_back(video_ref_());
  EXPECT_FALSE(h3::validate_reference_request(vids, lim, &err));

  // ... and the total, which binds before any single modality's does.
  std::vector<h3::MediaReference> mixed;
  for (int i = 0; i < 9; ++i) { mixed.push_back(image_ref_()); }
  for (int i = 0; i < 3; ++i) { mixed.push_back(video_ref_()); }
  EXPECT_TRUE((int)mixed.size() == 12);
  EXPECT_TRUE(h3::validate_reference_request(mixed, lim, &err));
  mixed.push_back(audio_ref_());
  EXPECT_FALSE(h3::validate_reference_request(mixed, lim, &err));
  std::printf("[minimax_h3_refenc] limits: %d images / %d videos / %d "
              "audios / %d total, audio never alone\n", lim.max_images,
              lim.max_videos, lim.max_audios, lim.max_references);
}

#endif  // VPIPE_BUILD_APPLE_SILICON
