#include "generative-models/minimax-h3/minimax-h3-references.h"

#include "apple-silicon/metal-compute/image-ops.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"

#include <algorithm>
#include <cmath>

namespace vpipe {
namespace genai {
namespace minimax_h3 {

namespace {

constexpr double kMinAspect = 1.0 / 4.0;
constexpr double kMaxAspect = 4.0;

std::uint8_t
clamp8_(double v)
{
  const double r = std::round(v);
  return (std::uint8_t)(r < 0.0 ? 0.0 : (r > 255.0 ? 255.0 : r));
}

}  // namespace

bool
resolve_reference_image_size(int height, int width, int short_edge,
                             int multiple, int* out_h, int* out_w)
{
  if (out_h == nullptr || out_w == nullptr) { return false; }
  if (height <= 0 || width <= 0 || short_edge <= 0 || multiple <= 0) {
    return false;
  }
  const double ratio = (double)width / (double)height;
  if (!(ratio >= kMinAspect && ratio <= kMaxAspect)) { return false; }
  // The short edge is scaled to `short_edge` and BOTH axes then round to
  // `multiple` -- there is no area cap here, unlike the target canvas, so
  // a large reference stays large and a small one is scaled up.
  const double scale = (double)short_edge / (double)std::min(height, width);
  const double m = (double)multiple;
  *out_h = (int)std::max(m, std::nearbyint(height * scale / m) * m);
  *out_w = (int)std::max(m, std::nearbyint(width * scale / m) * m);
  return true;
}

std::vector<int>
frame_resample_counts(int num_frames, double src_fps, double dst_fps)
{
  std::vector<int> counts;
  if (num_frames <= 0 || !(src_fps > 0.0) || !(dst_fps > 0.0)) {
    return counts;
  }
  counts.reserve((std::size_t)num_frames);
  if (src_fps == dst_fps) {
    counts.assign((std::size_t)num_frames, 1);
    return counts;
  }
  // ffmpeg's `fps` filter, which is what the reference reproduces: frame
  // i occupies the output slots [slot(i), slot(i+1)), where
  // slot(i) = floor(i * scale + 0.5), and the last frame is held to the
  // slot the stream's END rounds to. Whole frames only -- nothing is
  // blended, so a count of 0 is a DROPPED frame and a count above 1 is a
  // held one.
  const double scale = dst_fps / src_fps;
  auto slot = [&](double i) {
    return (long long)std::floor(i * scale + 0.5);
  };
  for (int i = 0; i < num_frames; ++i) {
    const long long a = slot((double)i);
    const long long b = (i + 1 < num_frames) ? slot((double)(i + 1))
                                             : slot((double)num_frames);
    counts.push_back((int)std::max<long long>(0, b - a));
  }
  return counts;
}

bool
resize_lanczos_planar_rgb(const std::uint8_t* rgb, int height, int width,
                          int out_h, int out_w,
                          std::vector<std::uint8_t>* out)
{
  if (rgb == nullptr || out == nullptr) { return false; }
  if (height <= 0 || width <= 0 || out_h <= 0 || out_w <= 0) { return false; }

  // The shared Pillow-exact coefficient builder, so this resize is the
  // same filter the image-ops GPU path and the diffusion conditioner
  // use rather than a fourth spelling of Lanczos-3.
  std::vector<int>   bx, by;
  std::vector<float> wx, wy;
  const int kx = metal_compute::build_lanczos_coeffs(
      width, 0.0, out_w, (double)width / (double)out_w, bx, wx);
  const int ky = metal_compute::build_lanczos_coeffs(
      height, 0.0, out_h, (double)height / (double)out_h, by, wy);
  if (kx <= 0 || ky <= 0) { return false; }

  out->assign((std::size_t)3 * out_h * out_w, 0);
  // Horizontal first, then vertical -- Pillow's order, and the
  // intermediate is rounded to U8 BETWEEN the passes because Pillow's
  // is: ImagingResampleHorizontal writes an 8-bit image that the
  // vertical pass then reads. Carrying float through is the nicer
  // filter and the wrong one -- it measured 354 of 1920 samples off by
  // up to 3 against PIL, where quantizing matches exactly.
  std::vector<std::uint8_t> mid((std::size_t)height * out_w, 0);
  for (int c = 0; c < 3; ++c) {
    const std::uint8_t* src = rgb + (std::size_t)c * height * width;
    for (int y = 0; y < height; ++y) {
      const std::uint8_t* row = src + (std::size_t)y * width;
      for (int x = 0; x < out_w; ++x) {
        const int b0 = bx[(std::size_t)x];
        double acc = 0.0;
        for (int t = 0; t < kx; ++t) {
          const int s = b0 + t;
          if (s < 0 || s >= width) { continue; }
          acc += (double)wx[(std::size_t)x * kx + t] * (double)row[s];
        }
        mid[(std::size_t)y * out_w + x] = clamp8_(acc);
      }
    }
    std::uint8_t* dst = out->data() + (std::size_t)c * out_h * out_w;
    for (int y = 0; y < out_h; ++y) {
      const int b0 = by[(std::size_t)y];
      for (int x = 0; x < out_w; ++x) {
        double acc = 0.0;
        for (int t = 0; t < ky; ++t) {
          const int s = b0 + t;
          if (s < 0 || s >= height) { continue; }
          acc += (double)wy[(std::size_t)y * ky + t] *
                 (double)mid[(std::size_t)s * out_w + x];
        }
        dst[(std::size_t)y * out_w + x] = clamp8_(acc);
      }
    }
  }
  return true;
}

bool
normalize_image_reference(const std::uint8_t* rgb, int height, int width,
                          int short_edge, int multiple,
                          std::vector<std::uint8_t>* out, int* out_h,
                          int* out_w, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (rgb == nullptr || out == nullptr || out_h == nullptr ||
      out_w == nullptr) {
    return fail("null argument");
  }
  int th = 0, tw = 0;
  if (!resolve_reference_image_size(height, width, short_edge, multiple, &th,
                                    &tw)) {
    return fail("a reference image must be within 1:4 and 4:1 and have a "
                "positive size");
  }
  *out_h = th;
  *out_w = tw;
  if (th == height && tw == width) {
    // Already there: copied through untouched, which is the only route
    // that reproduces the reference's pixels exactly.
    out->assign(rgb, rgb + (std::size_t)3 * height * width);
    return true;
  }
  if (!resize_lanczos_planar_rgb(rgb, height, width, th, tw, out)) {
    return fail("the reference image resize failed");
  }
  return true;
}

bool
normalize_video_reference(const std::uint8_t* frames, int num_frames,
                          int height, int width, double src_fps,
                          int target_frames, int multiple, int short_edge,
                          std::int64_t max_pixels,
                          std::vector<std::uint8_t>* out, int* out_frames,
                          int* out_h, int* out_w, double dst_fps,
                          std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (frames == nullptr || out == nullptr || out_frames == nullptr ||
      out_h == nullptr || out_w == nullptr) {
    return fail("null argument");
  }
  if (num_frames <= 0 || height <= 0 || width <= 0 || target_frames <= 0) {
    return fail("a reference video needs positive geometry");
  }
  if (!(src_fps > 0.0)) {
    return fail("a reference video must have a positive frame rate");
  }

  // 1. Onto the 24 fps grid, by holding and dropping WHOLE frames.
  const std::vector<int> counts =
      frame_resample_counts(num_frames, src_fps, dst_fps);
  if (counts.empty()) { return fail("the frame-rate resample failed"); }
  // 2. Truncated to the generated length. Built as an index list first
  // so the truncation costs nothing: a 15-second reference against a
  // 5-second target would otherwise materialize three times what it
  // needs and then throw two thirds away.
  std::vector<int> pick;
  pick.reserve((std::size_t)target_frames);
  for (int i = 0; i < num_frames && (int)pick.size() < target_frames; ++i) {
    for (int k = 0; k < counts[(std::size_t)i] &&
                    (int)pick.size() < target_frames; ++k) {
      pick.push_back(i);
    }
  }
  if (pick.empty()) { return fail("the reference video resampled to nothing"); }

  // 3. Onto the canvas its OWN aspect ratio resolves to -- the same rule
  // the target follows, unlike an image reference.
  int th = 0, tw = 0;
  if (!resolve_canvas_size((double)width, (double)height, multiple, short_edge,
                           max_pixels, &th, &tw)) {
    return fail("a reference video must be within 1:4 and 4:1");
  }
  *out_frames = (int)pick.size();
  *out_h = th;
  *out_w = tw;

  const std::size_t plane = (std::size_t)3 * th * tw;
  out->assign(plane * pick.size(), 0);
  const bool same = (th == height && tw == width);
  // A frame held more than once is resized ONCE and copied, which is
  // both faster and exactly equal -- the resize is deterministic.
  std::vector<std::uint8_t> scratch;
  int cached = -1;
  for (std::size_t o = 0; o < pick.size(); ++o) {
    const int src_i = pick[o];
    const std::uint8_t* src =
        frames + (std::size_t)src_i * 3 * height * width;
    if (same) {
      std::copy(src, src + plane, out->data() + o * plane);
      continue;
    }
    if (src_i != cached) {
      if (!resize_lanczos_planar_rgb(src, height, width, th, tw, &scratch)) {
        return fail("the reference video resize failed");
      }
      cached = src_i;
    }
    std::copy(scratch.begin(), scratch.end(), out->data() + o * plane);
  }
  return true;
}

bool
normalize_audio_reference(const float* pcm, int channels, int samples,
                          int sample_rate, double max_seconds,
                          std::vector<float>* out, int* out_samples,
                          std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (pcm == nullptr || out == nullptr) { return fail("null argument"); }
  if (channels != 1 && channels != 2) {
    return fail("a reference soundtrack must be mono or stereo, got " +
                std::to_string(channels) + " channels");
  }
  if (samples <= 0 || sample_rate <= 0 || !(max_seconds > 0.0)) {
    return fail("a reference soundtrack needs positive geometry");
  }
  // Truncated AT THE SOURCE RATE. The reference cuts the decoded
  // waveform and resamples once; cutting after a resample lands on a
  // different sample and drifts by the ratio of the two rates.
  const long long cap = (long long)(max_seconds * (double)sample_rate);
  const int n = (int)std::min<long long>(samples, std::max<long long>(0, cap));
  if (n <= 0) { return fail("the reference soundtrack truncated to nothing"); }

  out->assign((std::size_t)2 * n, 0.0f);
  for (int c = 0; c < 2; ++c) {
    // Mono is upmixed by REPEATING the channel: the VAE is mono and
    // takes the stereo pair as two batch items, so a zeroed second
    // channel would condition the right side on silence.
    const int src_c = (channels == 1) ? 0 : c;
    const float* src = pcm + (std::size_t)src_c * samples;
    std::copy(src, src + n, out->data() + (std::size_t)c * n);
  }
  if (out_samples != nullptr) { *out_samples = n; }
  return true;
}

}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe
