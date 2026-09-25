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
#include "generative-models/minimax-h3/minimax-h3-references.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
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

// The progress contract, without a single model loaded.
//
// Reference encoding is the longest stretch of a `ref2va` run that is
// not the denoise -- the conditioner is streamed and every reference is
// read twice -- so video-ref-encoder drives a progress bar off this
// callback. What makes the bar honest is the COUNT: `total` is one more
// than the number of references, because the presentation is a single
// conditioner call over the whole request and is routinely the longest
// thing in the phase. Reporting refs.size() would put the bar at 100%
// and leave it there for the part people actually wait through.
//
// Checkable with every model pointer null: the per-reference work is all
// guarded, so what is left is exactly the traversal and its reports.
//
// The bar is weighed in WORK, planned before anything runs. With no VAE
// attached a reference weighs one frame of its canvas, and that makes the
// plan checkable against what the loop then did: each reference's share
// has to equal the canvas the fit record says it was resized to. That is
// the property that matters -- the plan reads the same geometry rule as
// the resize, and a copy of the rule would drift.
// The prompt-only form gets through the whole encode, not just the
// limits check: a request with no references packs no rows and emits no
// layout entries, which is what makes the stage's sideband say
// `references: []` rather than something a consumer has to interpret.
//
// Driven with every model pointer null, as the progress test is, so it
// needs no GPU; what the CONDITIONER makes of an empty presentation is
// minimax_h3_text_enc.prompt_only_presentation_is_the_prompt.
TEST(minimax_h3_refenc, an_empty_request_encodes_to_no_rows)
{
  h3::ReferencePlan plan;
  plan.target_frames = 39;
  h3::ReferenceEncoders models;          // every pointer null on purpose
  std::vector<std::string> said;
  models.progress = [&said](std::uint64_t, std::uint64_t,
                            const std::string& detail) {
    said.push_back(detail);
  };

  h3::EncodedReferences enc;
  std::string err;
  const bool ok = h3::encode_references({}, "a prompt", plan, models, &enc,
                                        &err);
  if (!ok) { std::printf("[minimax_h3_refenc] %s\n", err.c_str()); }
  ASSERT_TRUE(ok);
  if (!ok) { return; }
  EXPECT_TRUE(enc.layout.empty());
  EXPECT_TRUE(enc.fits.empty());
  EXPECT_TRUE(enc.video_rows.empty());
  EXPECT_TRUE(enc.audio_rows.empty());
  // Finished, and said so -- a bar left short of "done" reads as a hang.
  EXPECT_TRUE(!said.empty() && said.back() == "done");
}

TEST(minimax_h3_refenc, progress_is_weighed_by_work_planned_up_front)
{
  h3::ReferencePlan plan;
  plan.target_frames     = 39;
  plan.canvas_short_edge = 0;            // the clip's own size

  std::vector<h3::MediaReference> refs;
  {
    // A still that asks for 128 on its short edge.
    h3::MediaReference m;
    m.kind       = h3::MediaReference::Kind::kImage;
    m.num_frames = 1;
    m.height     = 64;
    m.width      = 64;
    m.short_edge = 128;
    m.rgb.assign((std::size_t)3 * 64 * 64, 10);
    refs.push_back(std::move(m));
  }
  {
    // A 30 fps clip, longer than the target once on the 24 fps grid.
    h3::MediaReference m;
    m.kind       = h3::MediaReference::Kind::kVideo;
    m.num_frames = 60;
    m.height     = 96;
    m.width      = 160;
    m.fps        = 30.0;
    m.rgb.assign((std::size_t)60 * 3 * 96 * 160, 20);
    refs.push_back(std::move(m));
  }

  struct Seen {
    std::uint64_t done, total;
    std::string detail;
  };
  std::vector<Seen> seen;
  h3::ReferenceEncoders models;          // every pointer null on purpose
  models.progress = [&seen](std::uint64_t done, std::uint64_t total,
                            const std::string& detail) {
    seen.push_back({done, total, detail});
  };

  h3::EncodedReferences enc;
  std::string err;
  const bool ok = h3::encode_references(refs, "a prompt", plan, models,
                                        &enc, &err);
  std::printf("[minimax_h3_refenc] progress:");
  for (const auto& p : seen) {
    std::printf(" %llu/%llu '%s'", (unsigned long long)p.done,
                (unsigned long long)p.total, p.detail.c_str());
  }
  std::printf("%s%s\n", err.empty() ? "" : "  err: ", err.c_str());
  ASSERT_TRUE(ok);
  ASSERT_TRUE(enc.fits.size() == 2);
  ASSERT_TRUE(!seen.empty());
  if (!ok || enc.fits.size() != 2 || seen.empty()) { return; }

  const std::uint64_t w0 =
      (std::uint64_t)enc.fits[0].canvas_h * enc.fits[0].canvas_w;
  const std::uint64_t w1 =
      (std::uint64_t)enc.fits[1].canvas_h * enc.fits[1].canvas_w;
  EXPECT_TRUE(w0 == 128u * 128u);        // the still's own short edge won
  const std::uint64_t total = w0 + w1;

  // ONE total for the whole request, never rescaled, and the bar never
  // runs backwards. (No conditioner is attached, so no report here is
  // the indeterminate one.)
  std::uint64_t last = 0;
  bool steady = true, forward = true;
  for (const auto& p : seen) {
    steady = steady && p.total == total;
    forward = forward && p.done >= last;
    last = p.done;
  }
  EXPECT_TRUE(steady);
  EXPECT_TRUE(forward);
  EXPECT_TRUE(seen.front().done == 0);
  EXPECT_TRUE(seen.back().done == total && seen.back().detail == "done");

  // Each reference's share is its planned work: the bar stands at w0
  // when the second one starts, which is where the fit record says the
  // first one's canvas put it.
  bool second_started_at_w0 = false, named_phase = false;
  for (const auto& p : seen) {
    if (p.detail == "reference 2/2 (video)") {
      second_started_at_w0 = p.done == w0;
    }
    if (p.detail == "reference 1/2 (image): done") {
      EXPECT_TRUE(p.done == w0);
    }
    if (p.detail == "reference 2/2 (video): resize") { named_phase = true; }
  }
  EXPECT_TRUE(second_started_at_w0);
  EXPECT_TRUE(named_phase);
}

// A STOP REACHES THE REFERENCE ENCODE, and arrives as a stop.
//
// This is the longest stretch in an H3 graph that is not the denoise --
// the conditioner is loaded and streamed and every reference is read
// twice, so on a memory-bounded box it is minutes -- and it had no stop
// check anywhere. A stop request was served whenever the whole request
// happened to finish.
//
// TWO THINGS ARE UNDER TEST and the second is the one a caller depends
// on. That it stops at all, and that it says so by returning false with
// an EMPTY reason: the stage tells a stop from a failure by the absence
// of a message, so that it must not fill one in. A stop reported as a
// failure puts a warning in front of someone who pressed stop.
//
// Driven with every model pointer null, as the progress test is: that
// runs the planning and the phase structure with no GPU behind it, so
// the trigger can be placed exactly. The BOUND -- how long a stop
// actually takes on real work -- is not a thing this harness can see,
// and is measured end to end instead; it rests on the video VAE
// checking between tiles, each of which commits and waits.
TEST(minimax_h3_refenc, a_stop_is_honoured_and_is_not_a_failure)
{
  h3::ReferencePlan plan;
  plan.target_frames     = 39;
  plan.canvas_short_edge = 0;

  std::vector<h3::MediaReference> refs;
  {
    h3::MediaReference m;
    m.kind = h3::MediaReference::Kind::kImage;
    m.num_frames = 1; m.height = 64; m.width = 64; m.short_edge = 128;
    m.rgb.assign((std::size_t)3 * 64 * 64, 10);
    refs.push_back(std::move(m));
  }
  {
    h3::MediaReference m;
    m.kind = h3::MediaReference::Kind::kVideo;
    m.num_frames = 60; m.height = 96; m.width = 160; m.fps = 30.0;
    m.rgb.assign((std::size_t)60 * 3 * 96 * 160, 20);
    refs.push_back(std::move(m));
  }

  // Fire once the FIRST reference is finished, so the stop lands at a
  // boundary the encoder reaches mid-request rather than before it
  // starts -- a predicate that is true from the outset would be
  // answered by the first check and prove only that one check exists.
  bool stop_now = false;
  std::vector<std::string> seen;
  h3::ReferenceEncoders models;
  models.progress = [&](std::uint64_t, std::uint64_t,
                        const std::string& detail) {
    seen.push_back(detail);
    if (detail == "reference 1/2 (image): done") { stop_now = true; }
  };
  models.stopping = [&]() { return stop_now; };

  h3::EncodedReferences enc;
  std::string err;
  const bool ok = h3::encode_references(refs, "a prompt", plan, models,
                                        &enc, &err);

  EXPECT_FALSE(ok);
  // THE CONTRACT THE STAGE READS. An empty reason is the signal.
  EXPECT_TRUE(err.empty());
  // And it really did stop early: the second reference never ran a
  // phase, and the request never reported itself done.
  bool second_phase_ran = false, said_done = false;
  for (const std::string& d : seen) {
    if (d.rfind("reference 2/2", 0) == 0 && d.find(':') != std::string::npos) {
      second_phase_ran = true;
    }
    if (d == "done") { said_done = true; }
  }
  EXPECT_FALSE(second_phase_ran);
  EXPECT_FALSE(said_done);
  EXPECT_TRUE(enc.fits.size() < 2);
  std::printf("[minimax_h3_refenc] stop after reference 1: ok=%d err='%s' "
              "(%zu reports, second phase ran=%d, said done=%d)\n",
              (int)ok, err.c_str(), seen.size(), (int)second_phase_ran,
              (int)said_done);
}

// The plan's geometry IS the resize's: video_reference_geometry is what
// normalize_video_reference asks, so the two agree on every input --
// rate conversions that hold and drop, truncation, the never-upscale
// short edge, the area cap.
TEST(minimax_h3_refenc, planned_geometry_is_the_resize_geometry)
{
  struct Case {
    int frames, h, w;
    double fps;
    int target, edge;
    std::int64_t cap;
  };
  const Case cases[] = {
      {60, 96, 160, 30.0, 39, 0, 768LL * 1344},    // drop to 24, truncate
      {20, 96, 160, 12.0, 39, 0, 768LL * 1344},    // hold up to 24
      {30, 90, 160, 24.0, 21, 128, 768LL * 1344},  // upscale to 128
      {10, 720, 1280, 25.0, 39, 768, 256LL * 448}, // the cap binds
  };
  for (const Case& c : cases) {
    std::vector<std::uint8_t> rgb((std::size_t)c.frames * 3 * c.h * c.w, 5);
    std::vector<std::uint8_t> out;
    int nf = 0, th = 0, tw = 0, pf = 0, ph = 0, pw = 0;
    std::string e1, e2;
    const bool a = h3::normalize_video_reference(
        rgb.data(), c.frames, c.h, c.w, c.fps, c.target, 32, c.edge, c.cap,
        &out, &nf, &th, &tw, h3::kFps, &e1);
    const bool b = h3::video_reference_geometry(
        c.frames, c.h, c.w, c.fps, c.target, 32, c.edge, c.cap, &pf, &ph,
        &pw, h3::kFps, &e2);
    std::printf("[minimax_h3_refenc] %dx%d %df@%.0f -> %dx%d %df "
                "(planned %dx%d %df)\n", c.w, c.h, c.frames, c.fps, tw, th,
                nf, pw, ph, pf);
    EXPECT_TRUE(a && b);
    EXPECT_TRUE(nf == pf && th == ph && tw == pw);
    EXPECT_TRUE(out.size() == (std::size_t)nf * 3 * th * tw);
  }
}

// The fit record: three temporal reductions counted APART, and the
// per-reference short edge overriding the plan's.
//
// Kept separate because they have different remedies. A 30 fps clip
// loses a fifth of its frames to the RATE resample whether or not it is
// also too long; `frames` is what fixes the truncation and nothing fixes
// the resample. One "kept 43%" would tell a user something was lost
// without telling them which knob gets it back.
TEST(minimax_h3_refenc, fit_reports_each_reduction_separately)
{
  h3::ReferencePlan plan;
  plan.target_frames     = 39;
  plan.canvas_short_edge = 0;            // never upsample

  std::vector<h3::MediaReference> refs;
  {
    // 1920x1080 at 30 fps, 90 frames: over the area cap AND over length,
    // and at a rate that does not divide 24.
    h3::MediaReference m;
    m.kind       = h3::MediaReference::Kind::kVideo;
    m.num_frames = 90;
    m.height     = 1080;
    m.width      = 1920;
    m.fps        = 30.0;
    m.rgb.assign((std::size_t)90 * 3 * 1080 * 1920, 7);
    refs.push_back(std::move(m));
  }
  {
    // A still that states its own short edge, overriding the plan's.
    h3::MediaReference m;
    m.kind       = h3::MediaReference::Kind::kImage;
    m.num_frames = 1;
    m.height     = 64;
    m.width      = 64;
    m.short_edge = 128;
    m.rgb.assign((std::size_t)3 * 64 * 64, 9);
    refs.push_back(std::move(m));
  }

  h3::ReferenceEncoders models;          // every pointer null on purpose
  h3::EncodedReferences enc;
  std::string err;
  ASSERT_TRUE(h3::encode_references(refs, "p", plan, models, &enc, &err));
  ASSERT_TRUE(enc.fits.size() == 2);

  const auto& v = enc.fits[0];
  std::printf("[minimax_h3_refenc] clip %dx%d -> %dx%d, %d -> %d -> %d "
              "frames\n", v.src_w, v.src_h, v.canvas_w, v.canvas_h,
              v.src_frames, v.rate_frames, v.used_frames);
  // The cap binds, and never upward: both axes are below the source.
  EXPECT_TRUE(v.rescaled());
  EXPECT_FALSE(v.upscaled());
  EXPECT_TRUE(v.canvas_h <= 1080 && v.canvas_w <= 1920);
  EXPECT_TRUE((std::int64_t)v.canvas_h * v.canvas_w <= 768LL * 1344);
  // 90 frames at 30 fps is 72 at 24 -- the rate resample alone, before
  // any truncation. Then `target_frames` cuts 72 to 39.
  EXPECT_TRUE(v.rate_frames == 72);
  EXPECT_TRUE(v.used_frames == 39);
  EXPECT_TRUE(v.src_frames == 90);

  const auto& i = enc.fits[1];
  // The reference's own 128 won, not the plan's 2048 default.
  EXPECT_TRUE(i.canvas_h == 128 && i.canvas_w == 128);
  EXPECT_TRUE(i.upscaled());
  std::printf("[minimax_h3_refenc] still 64x64 -> %dx%d on its own short "
              "edge\n", i.canvas_w, i.canvas_h);
}

TEST(minimax_h3_refenc, request_limits)
{
  h3::ReferenceLimits lim;
  std::string err;

  // Empty is the prompt-only form, and passes: nothing in its
  // presentation describes a reference the model cannot see, which is
  // what the audio-alone rule below is about. Whether the caller MEANT
  // it is the stage's question.
  EXPECT_TRUE(h3::validate_reference_request({}, lim, &err));

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

// A CARRIED latent and a VAE-encoded one become DiT rows by two separate
// loops over the same layout: pack_latent_rows_ over an f32 latent that
// is already whitened, pack_condition_rows_ over the VAE's bf16 moments
// (mean half first) that it whitens on the way. The carried-latent route
// is only right if the two place every value identically, and two loops
// over one layout are exactly the shape of thing that drifts apart.
//
// Pinned two ways. With identity whitening and bf16-exact values the two
// must agree BIT FOR BIT, which is the layout claim with nothing else in
// it. With a real mean and std they must agree to the bf16 rounding of
// the moments -- which says the whitening runs in the direction a
// carried latent assumes ((m - mean) / std), not its inverse.
TEST(minimax_h3_refenc, carried_latent_rows_match_the_vae_rows)
{
  const int z = 3, T = 2, Hh = 4, Ww = 6, ph = 2, pw = 2;
  const std::size_t vox = (std::size_t)T * Hh * Ww;
  auto to_bf16 = [](float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
  };
  // A latent whose every value is distinct and bf16-exact, so a swapped
  // index cannot hide behind two equal values.
  std::vector<float> lat((std::size_t)z * vox);
  for (std::size_t i = 0; i < lat.size(); ++i) {
    lat[i] = (float)((int)(i % 97) - 48) * 0.0625f;
  }
  // Moments are [2z, T, H, W], MEAN half first; the logvar half is noise
  // the packer must not read.
  std::vector<std::uint16_t> mom((std::size_t)2 * z * vox);
  for (std::size_t i = 0; i < lat.size(); ++i) { mom[i] = to_bf16(lat[i]); }
  for (std::size_t i = lat.size(); i < mom.size(); ++i) {
    mom[i] = to_bf16(1000.0f + (float)i);
  }

  std::vector<float> a, b;
  h3::pack_latent_rows_for_test(lat.data(), z, T, Hh, Ww, ph, pw, &a);
  h3::pack_condition_rows_for_test(mom.data(), z, T, Hh, Ww, ph, pw,
                                   {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f},
                                   &b);
  // T * (H/ph) * (W/pw) rows of z*ph*pw.
  EXPECT_TRUE(a.size() == (std::size_t)T * (Hh / ph) * (Ww / pw) * z * ph *
                              pw);
  EXPECT_TRUE(a == b);

  // A real whitening: the moments hold latent * std + mean, so the
  // moments packer has to undo exactly that to land on the latent.
  const std::vector<float> mean = {0.5f, -1.25f, 2.0f};
  const std::vector<float> sd = {0.75f, 1.5f, 3.0f};
  for (int c = 0; c < z; ++c) {
    for (std::size_t v = 0; v < vox; ++v) {
      const std::size_t k = (std::size_t)c * vox + v;
      mom[k] = to_bf16(lat[k] * sd[(std::size_t)c] + mean[(std::size_t)c]);
    }
  }
  std::vector<float> w;
  h3::pack_condition_rows_for_test(mom.data(), z, T, Hh, Ww, ph, pw, mean,
                                   sd, &w);
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size() && i < w.size(); ++i) {
    worst = std::max(worst, (double)std::fabs(a[i] - w[i]));
  }
  // bf16 keeps 8 bits: |m| here is at most ~11, so one ulp is 0.0625,
  // and dividing by std >= 0.75 grows it to ~0.083 at the worst.
  std::printf("[minimax_h3_refenc] carried vs VAE rows: identity whitening "
              "%s; real whitening max |diff| %.4f\n",
              a == b ? "bit-identical" : "DIFFERS", worst);
  EXPECT_TRUE(w.size() == a.size());
  EXPECT_TRUE(worst < 0.1);
}
