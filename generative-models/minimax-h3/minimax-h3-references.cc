#include "generative-models/minimax-h3/minimax-h3-references.h"

#include "apple-silicon/metal-compute/image-ops.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"

#include <algorithm>
#include <cmath>
#include <cstring>

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
resolve_canvas_within(int height, int width, int multiple,
                      std::int64_t max_pixels, int* out_h, int* out_w)
{
  if (out_h == nullptr || out_w == nullptr) { return false; }
  if (height <= 0 || width <= 0 || multiple <= 0) { return false; }
  const double ratio = (double)width / (double)height;
  if (!(ratio >= kMinAspect && ratio <= kMaxAspect)) { return false; }
  double h = (double)height, w = (double)width;
  const double area = h * w;
  if (max_pixels > 0 && area > (double)max_pixels) {
    const double s = std::sqrt((double)max_pixels / area);
    h *= s;
    w *= s;
  }
  // FLOOR, not nearest: nearest is above as often as below, and one grid
  // line up is still an upsample.
  const double m = (double)multiple;
  *out_h = (int)std::max(m, std::floor(h / m) * m);
  *out_w = (int)std::max(m, std::floor(w / m) * m);
  return true;
}

bool
resolve_reference_image_size(int height, int width, int short_edge,
                             int multiple, std::int64_t max_pixels,
                             int* out_h, int* out_w)
{
  if (short_edge == 0) {
    return resolve_canvas_within(height, width, multiple, max_pixels, out_h,
                                 out_w);
  }
  if (out_h == nullptr || out_w == nullptr) { return false; }
  if (height <= 0 || width <= 0 || short_edge < 0 || multiple <= 0) {
    return false;
  }
  const double ratio = (double)width / (double)height;
  if (!(ratio >= kMinAspect && ratio <= kMaxAspect)) { return false; }
  // The short edge is scaled to `short_edge` and BOTH axes then round to
  // `multiple`. `max_pixels` is 0 for the released checkpoint -- an image
  // reference has no area cap there, so a large one stays large and a
  // small one is scaled up.
  const double scale = (double)short_edge / (double)std::min(height, width);
  double h = (double)height * scale, w = (double)width * scale;
  const double area = h * w;
  if (max_pixels > 0 && area > (double)max_pixels) {
    const double s = std::sqrt((double)max_pixels / area);
    h *= s;
    w *= s;
  }
  const double m = (double)multiple;
  *out_h = (int)std::max(m, std::nearbyint(h / m) * m);
  *out_w = (int)std::max(m, std::nearbyint(w / m) * m);
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
prepare_planar_resize(int src_h, int src_w, int dst_h, int dst_w,
                      PlanarResize* out)
{
  if (out == nullptr) { return false; }
  *out = PlanarResize{};
  if (src_h <= 0 || src_w <= 0 || dst_h <= 0 || dst_w <= 0) { return false; }

  // The shared Pillow-exact coefficient builder, so this resize is the
  // same filter the image-ops GPU path and the diffusion conditioner
  // use rather than a fourth spelling of Lanczos-3.
  out->kx = metal_compute::build_lanczos_coeffs(
      src_w, 0.0, dst_w, (double)src_w / (double)dst_w, out->bx, out->wx);
  out->ky = metal_compute::build_lanczos_coeffs(
      src_h, 0.0, dst_h, (double)src_h / (double)dst_h, out->by, out->wy);
  if (out->kx <= 0 || out->ky <= 0) { return false; }
  out->src_h = src_h; out->src_w = src_w;
  out->dst_h = dst_h; out->dst_w = dst_w;

  // Taps inside the source, per output column and row. The builder
  // clamps its first index to 0 and zeroes the weight of every tap past
  // the far edge, so this stops exactly where the old per-tap `continue`
  // did -- and the `s < 0` half of that test was never reachable.
  auto taps = [](const std::vector<int>& b, int k, int extent,
                 std::vector<int>* n) {
    n->resize(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) {
      const int room = extent - b[i];
      (*n)[i] = room < 0 ? 0 : (room < k ? room : k);
    }
  };
  taps(out->bx, out->kx, src_w, &out->nx);
  taps(out->by, out->ky, src_h, &out->ny);

  out->mid.resize((std::size_t)src_h * dst_w);
  out->acc.resize((std::size_t)dst_w);
  return true;
}

bool
run_planar_resize(PlanarResize& p, const std::uint8_t* rgb, std::uint8_t* out)
{
  if (rgb == nullptr || out == nullptr || p.empty()) { return false; }
  const int H = p.src_h, W = p.src_w, OH = p.dst_h, OW = p.dst_w;
  // Horizontal first, then vertical -- Pillow's order, and the
  // intermediate is rounded to U8 BETWEEN the passes because Pillow's
  // is: ImagingResampleHorizontal writes an 8-bit image that the
  // vertical pass then reads. Carrying float through is the nicer
  // filter and the wrong one -- it measured 354 of 1920 samples off by
  // up to 3 against PIL, where quantizing matches exactly.
  for (int c = 0; c < 3; ++c) {
    const std::uint8_t* src = rgb + (std::size_t)c * H * W;
    for (int y = 0; y < H; ++y) {
      const std::uint8_t* row = src + (std::size_t)y * W;
      std::uint8_t* mrow = p.mid.data() + (std::size_t)y * OW;
      for (int x = 0; x < OW; ++x) {
        const std::uint8_t* sp = row + p.bx[(std::size_t)x];
        const float* wp = &p.wx[(std::size_t)x * p.kx];
        const int n = p.nx[(std::size_t)x];
        double acc = 0.0;
        for (int t = 0; t < n; ++t) { acc += (double)wp[t] * (double)sp[t]; }
        mrow[x] = clamp8_(acc);
      }
    }
    std::uint8_t* dst = out + (std::size_t)c * OH * OW;
    for (int y = 0; y < OH; ++y) {
      // Tap-outer, so the inner loop walks a whole intermediate ROW and
      // vectorizes. Each output column still accumulates its taps in
      // ascending order, which is what keeps this bit-identical to the
      // column-at-a-time spelling rather than merely close to it.
      const float* wp = &p.wy[(std::size_t)y * p.ky];
      const std::uint8_t* base = p.mid.data() + (std::size_t)p.by[y] * OW;
      const int n = p.ny[(std::size_t)y];
      double* a = p.acc.data();
      for (int x = 0; x < OW; ++x) { a[x] = 0.0; }
      for (int t = 0; t < n; ++t) {
        const double w = (double)wp[t];
        const std::uint8_t* mp = base + (std::size_t)t * OW;
        for (int x = 0; x < OW; ++x) { a[x] += w * (double)mp[x]; }
      }
      std::uint8_t* orow = dst + (std::size_t)y * OW;
      for (int x = 0; x < OW; ++x) { orow[x] = clamp8_(a[x]); }
    }
  }
  return true;
}

bool
resize_lanczos_planar_rgb(const std::uint8_t* rgb, int height, int width,
                          int out_h, int out_w,
                          std::vector<std::uint8_t>* out)
{
  if (rgb == nullptr || out == nullptr) { return false; }
  PlanarResize p;
  if (!prepare_planar_resize(height, width, out_h, out_w, &p)) {
    return false;
  }
  // `resize`, not `assign(n, 0)`: every byte below is written, so the
  // fill was only ever a memset the filter then overwrote.
  out->resize((std::size_t)3 * out_h * out_w);
  return run_planar_resize(p, rgb, out->data());
}

bool
normalize_image_reference(const std::uint8_t* rgb, int height, int width,
                          int short_edge, int multiple,
                          std::int64_t max_pixels,
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
  if (!resolve_reference_image_size(height, width, short_edge, multiple,
                                    max_pixels, &th, &tw)) {
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
  // the target follows, unlike an image reference. A `short_edge` of 0
  // asks for the SOURCE'S own, which is how a caller declines the
  // upscale; everything after it -- the aspect resolve, the area cap,
  // the rounding -- is the same rule either way, and the cap is still
  // what binds a source larger than the canvas.
  int th = 0, tw = 0;
  const bool sized =
      short_edge > 0
          ? resolve_canvas_size((double)width, (double)height, multiple,
                                short_edge, max_pixels, &th, &tw)
          : resolve_canvas_within(height, width, multiple, max_pixels, &th,
                                  &tw);
  if (!sized) {
    return fail("a reference video must be within 1:4 and 4:1");
  }
  *out_frames = (int)pick.size();
  *out_h = th;
  *out_w = tw;

  const std::size_t plane = (std::size_t)3 * th * tw;
  // `resize`, not `assign(n, 0)`: every byte is written below either
  // way, so the fill was a memset of the whole clip that the frames then
  // overwrote -- 121 MB of it on a 39-frame 1344x768 reference.
  out->resize(plane * pick.size());
  if (th == height && tw == width) {
    for (std::size_t o = 0; o < pick.size(); ++o) {
      std::memcpy(out->data() + o * plane,
                  frames + (std::size_t)pick[o] * plane, plane);
    }
    return true;
  }

  // ONE plan for the whole clip. Every frame shares a geometry, and the
  // coefficient tables, the u8 intermediate and the tap bounds are
  // properties of that geometry rather than of a frame. MEASURED on a
  // 39-frame 960x544 -> 1344x768 clip, 507 ms before and 350 after, with
  // the same bytes out: ~19 ms of it was rebuilding the tables, and most
  // of the rest was re-deriving the tap bounds inside the innermost
  // loop.
  PlanarResize plan;
  if (!prepare_planar_resize(height, width, th, tw, &plan)) {
    return fail("the reference video resize failed");
  }
  const std::size_t src_plane = (std::size_t)3 * height * width;
  // `pick` is non-decreasing, so a frame held more than once occupies a
  // RUN of consecutive slots: resize into the run's first slot and copy
  // it forward from there. Straight into the destination -- the staging
  // buffer the copy used to come from was a second full-frame write per
  // unique frame, and the resize is deterministic, so the copy and a
  // re-resize are the same bytes.
  int cached = -1;
  std::size_t cached_at = 0;
  for (std::size_t o = 0; o < pick.size(); ++o) {
    std::uint8_t* dst = out->data() + o * plane;
    if (pick[o] == cached) {
      std::memcpy(dst, out->data() + cached_at * plane, plane);
      continue;
    }
    if (!run_planar_resize(plan, frames + (std::size_t)pick[o] * src_plane,
                           dst)) {
      return fail("the reference video resize failed");
    }
    cached    = pick[o];
    cached_at = o;
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
