#include "generative-models/magi2/magi2-preview.h"

#include "common/vpipe-format.h"
#include "generative-models/krea2/flow-sampler.h"
#include "generative-models/shared/torch-randn.h"

namespace vpipe {
namespace genai {
namespace magi2 {

#pragma clang fp contract(off)

void
guide(std::vector<float>& cond, const std::vector<float>& uncond, float g)
{
  for (std::size_t i = 0; i < cond.size(); ++i) {
    const float d = cond[i] - uncond[i];
    const float s = g * d;
    cond[i] = uncond[i] + s;
  }
}

bool
run_preview(MetalMagi2Dit& dit, const PreviewRequest& r,
            std::vector<float>* video, std::vector<float>* audio,
            std::string* err)
{
  return run_preview(
      [&dit](const std::vector<const MetalMagi2Dit::Sample*>& b,
             std::vector<std::vector<float>>* vv,
             std::vector<std::vector<float>>* va, std::string* e) {
        return dit.forward(b, vv, va, e);
      },
      dit.config().video_c, dit.config().audio_c, r, video, audio, err);
}

bool
run_preview(const BatchForward& forward, int video_c, int audio_c,
            const PreviewRequest& r, std::vector<float>* video,
            std::vector<float>* audio, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "MAGI-2 preview: " + m; }
    return false;
  };
  if (r.T <= 0 || r.h <= 0 || r.w <= 0 || r.audio_len < 0 || r.steps <= 0) {
    return fail("an empty request");
  }
  if (r.cond == nullptr || r.cond_rows <= 0) {
    return fail("no conditioning rows");
  }
  const bool guided_v = r.video_guidance != 1.0;
  const bool guided_a = r.audio_guidance != 1.0 && r.audio_len > 0;
  const bool guided = guided_v || guided_a;
  if (guided && (r.neg == nullptr || r.neg_rows <= 0)) {
    return fail("guidance asks for a negative condition and none was "
                "given");
  }
  const std::size_t nv = (std::size_t)video_c * r.T * r.h * r.w;
  const std::size_t na = (std::size_t)r.audio_len * audio_c;
  // The noise: video, then audio, from one generator (the release's two
  // torch.randn calls in turn).
  if (r.video_noise != nullptr) {
    video->assign(r.video_noise, r.video_noise + nv);
    audio->assign(r.audio_noise != nullptr ? r.audio_noise : r.video_noise,
                  (r.audio_noise != nullptr ? r.audio_noise
                                            : r.video_noise) + na);
  } else {
    TorchCpuRandn g(r.seed);
    *video = g.randn(nv);
    *audio = na > 0 ? g.randn(na) : std::vector<float>();
    if (video->size() != nv || audio->size() != na) {
      return fail("a latent too small for the generator's draw");
    }
  }
  FlowSampler sv(sampler_spec(), scheduler_spec(r.steps, r.shift));
  FlowSampler sa(sampler_spec(), scheduler_spec(r.steps, r.shift));
  sv.reset();
  sa.reset();
  const std::vector<double>& sig = sv.sigmas();

  MetalMagi2Dit::Sample cs, us;
  for (MetalMagi2Dit::Sample* s : {&cs, &us}) {
    s->T = r.T;
    s->H = r.h;
    s->W = r.w;
    s->audio_len = r.audio_len;
    s->images = r.images;
  }
  cs.text = r.cond;
  cs.text_len = r.cond_rows;
  us.text = r.neg;
  us.text_len = r.neg_rows;
  std::vector<const MetalMagi2Dit::Sample*> batch{&cs};
  if (guided) { batch.push_back(&us); }

  std::vector<std::vector<float>> vv, va;
  for (int i = 0; i < r.steps; ++i) {
    const float t = model_t(sig[(std::size_t)i]);
    for (MetalMagi2Dit::Sample* s : {&cs, &us}) {
      s->video = video->data();
      s->audio = audio->data();
      s->t = t;
      s->step = i;
    }
    if (!forward(batch, &vv, &va, err)) { return false; }
    if (guided_v) {
      guide(vv[0], vv[1], (float)r.video_guidance);
    }
    if (guided_a) {
      guide(va[0], va[1], (float)r.audio_guidance);
    }
    // UniPC asks for the velocity once a step, at the step's own sample;
    // the batch above computed it.
    int asked = 0;
    sv.step(i, *video, [&](const std::vector<float>&, double) {
      ++asked;
      return vv[0];
    });
    if (r.audio_len > 0) {
      sa.step(i, *audio, [&](const std::vector<float>&, double) {
        ++asked;
        return va[0];
      });
    }
    if (asked != (r.audio_len > 0 ? 2 : 1)) {
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
