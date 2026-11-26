#include "generative-models/magi2/magi2-refine.h"

#include "common/vpipe-format.h"
#include "generative-models/krea2/flow-sampler.h"
#include "generative-models/magi2/magi2-preview.h"
#include "generative-models/magi2/magi2-schedule.h"
#include "generative-models/shared/torch-randn.h"

#include <cmath>

namespace vpipe {
namespace genai {
namespace magi2 {

#pragma clang fp contract(off)

float
refine_signal_scale(int index)
{
  // betas = linspace(sqrt(0.00085), sqrt(0.012), 1000, f64) ** 2, their
  // alphas' cumulative product in f64, then f32: the square roots
  // rescaled so the last is 0 and the first unchanged (zero SNR), all in
  // f32 as torch does it. The flips cancel.
  constexpr int N = 1000;
  const double a = std::sqrt(0.00085), b = std::sqrt(0.012);
  std::vector<double> acp((std::size_t)N);
  double p = 1.0;
  for (int i = 0; i < N; ++i) {
    // torch.linspace(start, end, steps) in f64: start + i * step for the
    // first half, end - (steps - 1 - i) * step for the rest.
    const double step = (b - a) / (double)(N - 1);
    const double x = i < N / 2 ? a + step * (double)i
                               : b - step * (double)(N - 1 - i);
    p *= 1.0 - x * x;
    acp[(std::size_t)i] = p;
  }
  if (index < 0 || index >= N) { return 0.0f; }
  const float s0 = std::sqrt((float)acp[0]);
  const float st = std::sqrt((float)acp[(std::size_t)N - 1]);
  float v = std::sqrt((float)acp[(std::size_t)index]);
  v -= st;
  v *= s0 / (s0 - st);
  return v;
}

std::vector<float>
upsample_trilinear(const float* x, int C, int T, int H, int W, int To,
                   int Ho, int Wo)
{
  std::vector<float> out((std::size_t)C * To * Ho * Wo);
  // ATen's area_pixel_compute_source_index with align_corners: the scale
  // (in - 1) / (out - 1) in f32, the source index scale * o, its integer
  // part and the next index (clamped), lambdas l1 = src - i0, l0 = 1 - l1.
  struct Tap {
    int i0, i1;
    float l0, l1;
  };
  auto taps = [](int in, int outn) {
    std::vector<Tap> v((std::size_t)outn);
    const float scale =
        outn > 1 ? (float)(in - 1) / (float)(outn - 1) : 0.0f;
    for (int o = 0; o < outn; ++o) {
      const float src = scale * (float)o;
      const int i0 = (int)src;
      const int ip = i0 < in - 1 ? 1 : 0;
      const float l1 = src - (float)i0;
      v[(std::size_t)o] = {i0, i0 + ip, 1.0f - l1, l1};
    }
    return v;
  };
  const std::vector<Tap> tt = taps(T, To), th = taps(H, Ho), tw = taps(W, Wo);
  for (int c = 0; c < C; ++c) {
    const float* xc = x + (std::size_t)c * T * H * W;
    float* oc = out.data() + (std::size_t)c * To * Ho * Wo;
    for (int t = 0; t < To; ++t) {
      const Tap& a = tt[(std::size_t)t];
      for (int h = 0; h < Ho; ++h) {
        const Tap& b = th[(std::size_t)h];
        for (int w = 0; w < Wo; ++w) {
          const Tap& d = tw[(std::size_t)w];
          auto at = [&](int ti, int hi, int wi) {
            return xc[((std::size_t)ti * H + hi) * W + wi];
          };
          const float v =
              a.l0 * (b.l0 * (d.l0 * at(a.i0, b.i0, d.i0) +
                              d.l1 * at(a.i0, b.i0, d.i1)) +
                      b.l1 * (d.l0 * at(a.i0, b.i1, d.i0) +
                              d.l1 * at(a.i0, b.i1, d.i1))) +
              a.l1 * (b.l0 * (d.l0 * at(a.i1, b.i0, d.i0) +
                              d.l1 * at(a.i1, b.i0, d.i1)) +
                      b.l1 * (d.l0 * at(a.i1, b.i1, d.i0) +
                              d.l1 * at(a.i1, b.i1, d.i1)));
          oc[((std::size_t)t * Ho + h) * Wo + w] = v;
        }
      }
    }
  }
  return out;
}

bool
run_refine(MetalMagi2Refiner& refiner, const RefineRequest& r,
           std::vector<float>* video, std::string* err)
{
  return run_refine(
      [&refiner](const MetalMagi2Refiner::Sample& s, std::vector<float>* v,
                 std::string* e) { return refiner.forward(s, v, e); },
      refiner.config().video_c, r, video, err);
}

bool
run_refine(const RefineForward& forward, int video_c, const RefineRequest& r,
           std::vector<float>* video, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "MAGI-2 refine: " + m; }
    return false;
  };
  if (r.T <= 0 || r.h <= 0 || r.w <= 0 || r.latent == nullptr ||
      r.H <= 0 || r.W <= 0 || r.steps <= 0) {
    return fail("an empty request");
  }
  if (r.cond == nullptr || r.cond_rows <= 0) {
    return fail("no conditioning rows");
  }
  const bool guided = r.guidance != 1.0;
  if (guided && (r.neg == nullptr || r.neg_rows <= 0)) {
    return fail("guidance asks for a negative condition and none was "
                "given");
  }
  const int To = refine_latent_frames(r.T);
  *video = upsample_trilinear(r.latent, video_c, r.T, r.h, r.w, To, r.H,
                              r.W);
  // Re-noised: x s + n sqrt(1 - s^2), each product and the sum rounded.
  {
    const float s = refine_signal_scale(kRefineNoiseIndex);
    const float c = std::sqrt(1.0f - s * s);
    std::vector<float> drawn;
    const float* n = r.noise;
    if (n == nullptr) {
      TorchCpuRandn g(r.seed);
      drawn = g.randn(video->size());
      if (drawn.size() != video->size()) {
        return fail("a latent too large for the generator's draw");
      }
      n = drawn.data();
    }
    for (std::size_t i = 0; i < video->size(); ++i) {
      const float a = (*video)[i] * s;
      const float b = n[i] * c;
      (*video)[i] = a + b;
    }
  }
  FlowSampler sv(sampler_spec(), scheduler_spec(r.steps, r.shift));
  sv.reset();
  MetalMagi2Refiner::Sample cs, us;
  for (MetalMagi2Refiner::Sample* s : {&cs, &us}) {
    s->T = To;
    s->H = r.H;
    s->W = r.W;
  }
  cs.text = r.cond;
  cs.text_len = r.cond_rows;
  us.text = r.neg;
  us.text_len = r.neg_rows;
  std::vector<float> vc, vu;
  for (int i = 0; i < r.steps; ++i) {
    cs.video = video->data();
    us.video = video->data();
    cs.step = i;
    us.step = i;
    if (!forward(cs, &vc, err)) { return false; }
    if (guided) {
      if (!forward(us, &vu, err)) { return false; }
      guide(vc, vu, (float)r.guidance);
    }
    int asked = 0;
    sv.step(i, *video, [&](const std::vector<float>&, double) {
      ++asked;
      return vc;
    });
    if (asked != 1) {
      return fail(fmt("the sampler asked for {} velocities at step {}",
                      asked, i)());
    }
    if (r.progress && !r.progress(i + 1, r.steps)) {
      if (err != nullptr) { err->clear(); }
      return false;
    }
  }
  return true;
}

}  // namespace magi2
}  // namespace genai
}  // namespace vpipe
