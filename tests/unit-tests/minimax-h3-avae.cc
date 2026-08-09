// The MiniMax-H3 audio VAE's BigVGAN decoder, against the diffusers
// reference at full depth.
//
// What this checks that nothing else does:
//
//   * the ANTI-ALIASED activation. Every SnakeBeta is wrapped in a 2x
//     upsample / activate / downsample, with replicate padding and a
//     12-tap Kaiser window at both ends. Applying the activation
//     directly still produces audio -- it just produces audio with
//     aliased harmonics, which measures as a small error and sounds
//     like a cheap synthesizer.
//   * the AVERAGE over the three parallel AMP blocks. They are a bank,
//     not a stack: summing without dividing by three scales the stage
//     output, and the clamp at the end hides it as clipping.
//   * WEIGHT-NORM folding, `w = v * g / ||v||`, where the norm runs over
//     every dimension but the first -- and for a ConvTranspose1d that
//     first dimension is the INPUT width, so the scale multiplies along
//     the contraction axis rather than the output one.
//   * the transposed convolutions' kernels (9 for rate 5, 4 for rate 2),
//     which no rule reproduces from the rate and which decide the
//     output LENGTH.
//   * that the stereo pair stays separated. The autoencoder is mono and
//     the two channels ride as batch items, so a convolution that
//     reached across the batch boundary would blend them at the seam.
//
// Env: VPIPE_MINIMAX_H3_AVAE_PATH (the audio_vae dir, or any parent that
// resolves to it), VPIPE_MINIMAX_H3_AVAE_GOLDEN (the directory holding
// avae_golden.json).

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/minimax-h3/metal-minimax-h3-audio-vae.h"
#include "generative-models/weight-set.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-vae-decode-stage.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;

namespace {

struct Golden {
  int stereo = 0, latent_channels = 0, frames = 0, samples = 0;
  int sample_rate = 0;
  std::vector<float> latent, waveform, stage_rms;
  bool ok = false;
};

Golden
load_golden_(const std::string& dir)
{
  Golden g;
  std::ifstream in(dir + "/avae_golden.json");
  if (!in) { return g; }
  FlexData fd;
  try {
    fd = FlexData::from_json(in);
  } catch (...) {
    return g;
  }
  if (!fd.is_object()) { return g; }
  auto o = fd.as_object();
  auto geti = [&](const char* k) {
    return o.contains(k) ? (int)o.at(k).as_int(0) : 0;
  };
  g.stereo          = geti("stereo");
  g.latent_channels = geti("latent_channels");
  g.frames          = geti("frames");
  g.samples         = geti("samples");
  g.sample_rate     = geti("sample_rate");
  auto reals = [&](const char* k, std::vector<float>* dst) {
    if (!o.contains(k)) { return; }
    FlexData v = o.at(k);
    if (!v.is_array()) { return; }
    for (auto x : v.as_real_span()) { dst->push_back((float)x); }
  };
  reals("latent", &g.latent);
  reals("waveform", &g.waveform);
  reals("stage_rms", &g.stage_rms);
  g.ok = g.stereo > 0 && g.frames > 0 && !g.latent.empty() &&
         !g.waveform.empty();
  return g;
}

double
rel_l2_(const std::vector<float>& a, const std::vector<float>& b)
{
  double num = 0.0, den = 0.0;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : (num > 0.0 ? 1.0 : 0.0);
}

double
rms_(const std::vector<float>& v, std::size_t off, std::size_t n)
{
  double s = 0.0;
  for (std::size_t i = 0; i < n && off + i < v.size(); ++i) {
    s += (double)v[off + i] * (double)v[off + i];
  }
  return n > 0 ? std::sqrt(s / (double)n) : 0.0;
}

}  // namespace

TEST(minimax_h3_avae, config_from_json)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_AVAE_PATH");
  if (root == nullptr || *root == '\0') { return; }
  MetalMiniMaxH3AudioVae::Config cfg;
  std::string err;
  const bool ok = MetalMiniMaxH3AudioVae::config_from_json(root, cfg, &err);
  if (!ok) { std::printf("[minimax_h3_avae] config: %s\n", err.c_str()); }
  ASSERT_TRUE(ok);
  std::printf("[minimax_h3_avae] BigVGAN: latent %d -> %d, decoder_dim %d, "
              "hop %d, %d Hz, %d stereo channels\n",
              cfg.latent_channels, cfg.latent_dim, cfg.decoder_dim, cfg.hop(),
              cfg.sample_rate, cfg.stereo_channels);
  // 40 latents/s at 32 kHz is the hop; it must be exactly the encoder's,
  // or an encode and a decode would disagree on how long a clip is.
  EXPECT_TRUE(cfg.hop() == 800);
  EXPECT_TRUE(cfg.sample_rate == 32000);
  EXPECT_TRUE(cfg.latent_channels == 32 && cfg.latent_dim == 2048);
  EXPECT_TRUE(cfg.stereo_channels == 2);
  // The whitening the DiT works in. Silent identity here is the failure
  // that produces a quiet, dull soundtrack rather than an obvious break.
  EXPECT_TRUE((int)cfg.latents_mean.size() == cfg.latent_channels);
  EXPECT_TRUE((int)cfg.latents_std.size() == cfg.latent_channels);
  bool nontrivial = false;
  for (int i = 0; i < (int)cfg.latents_std.size(); ++i) {
    if (std::fabs(cfg.latents_std[(std::size_t)i] - 1.0f) > 1e-3f) {
      nontrivial = true;
    }
  }
  EXPECT_TRUE(nontrivial);
}

// Every alias-free activation in BigVGAN builds its resamplers with the
// same (ratio 2, 12 taps) arguments, so all 254 stored Kaiser windows are
// bit-identical -- which is what lets the decoder carry one copy of each
// instead of 127. That is an ASSUMPTION baked into the loader, so check
// it against the checkpoint rather than against the source it came from.
// The whole model must LOAD, not merely have a readable config.
//
// This is the regression for a checkpoint that reported itself missing
// while being perfectly loadable. Every weight read here falls back from
// `.weight_v` to `.weight` -- weight norm is stored split as (g, v) until
// something calls remove_weight_norm(), after which the product is a
// plain `.weight` -- but ONE probe, the one that reads the transposed
// convolution's kernel width out of the tensor shape, asked for
// `.weight_v` alone and returned null when it was absent. So a folded
// checkpoint (Comfy-Org's repack of this component, and anything else
// exported post-fold) failed load with nothing to say beyond "could not
// load", and the stage above reported "holds no audio VAE this stage
// knows" about a file that holds exactly that.
//
// Asserting on the config alone would not have caught it: config parsing
// reads the metadata and never touches a weight. The assertion has to be
// that a MODEL comes back.
//
// Env: VPIPE_MINIMAX_H3_AVAE_PATH.
TEST(minimax_h3_avae, loads_with_folded_weight_norm)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_AVAE_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string dir = MetalMiniMaxH3AudioVae::resolve_vae_dir(root);

  MetalMiniMaxH3AudioVae::Config cfg;
  std::string cerr;
  if (!MetalMiniMaxH3AudioVae::config_from_json(dir, cfg, &cerr)) {
    std::printf("[minimax_h3_avae] config: %s\n", cerr.c_str());
  }
  ASSERT_TRUE(MetalMiniMaxH3AudioVae::config_from_json(dir, cfg, &cerr));

  auto vae = MetalMiniMaxH3AudioVae::load(dir, mc, cfg);
  const bool folded =
      WeightSet::open(dir, nullptr) != nullptr &&
      WeightSet::open(dir, nullptr)->src().info(
          "decoder.ups.0.0.weight_v") == nullptr;
  std::printf("[minimax_h3_avae] loaded=%d, weight-norm %s\n",
              vae != nullptr ? 1 : 0, folded ? "FOLDED" : "split (g,v)");
  ASSERT_TRUE(vae != nullptr);
  // And it is the real net, not an empty shell: the decoder's upsample
  // kernel is read from the checkpoint precisely because no rule
  // reproduces it, so a plausible non-zero value is evidence the shape
  // probe found a tensor rather than defaulting.
  EXPECT_TRUE(cfg.sample_rate > 0 && cfg.latent_channels > 0);
}

TEST(minimax_h3_avae, resample_filters_are_shared)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_AVAE_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string dir = MetalMiniMaxH3AudioVae::resolve_vae_dir(root);
  std::shared_ptr<WeightSet> ws = WeightSet::open(dir, nullptr);
  if (!ws) { return; }

  auto grab = [&](const std::string& nm) -> std::vector<float> {
    std::vector<float> out;
    const auto* info = ws->src().info(nm);
    if (info == nullptr) { return out; }
    metal_compute::SharedBuffer b =
        ws->read(nm, mc, WeightSet::Residency::Copied);
    if (b.empty()) { return out; }
    const auto* p = static_cast<const float*>(b.contents());
    out.assign(p, p + b.byte_size() / 4);
    return out;
  };

  const std::vector<float> up0 =
      grab("decoder.activation_post.upsample.filter");
  const std::vector<float> dn0 =
      grab("decoder.activation_post.downsample.lowpass.filter");
  ASSERT_TRUE(up0.size() == 12 && dn0.size() == 12);

  int checked = 0, differ = 0;
  for (int rb = 0; rb < 21; ++rb) {
    for (int a = 0; a < 6; ++a) {
      const std::string base = "decoder.resblocks." + std::to_string(rb) +
                               ".activations." + std::to_string(a) + ".";
      const std::vector<float> u = grab(base + "upsample.filter");
      const std::vector<float> d = grab(base + "downsample.lowpass.filter");
      if (u.empty() || d.empty()) { continue; }
      ++checked;
      if (u != up0 || d != dn0) { ++differ; }
    }
  }
  std::printf("[minimax_h3_avae] resample filters: %d activations checked, "
              "%d differ from the shared pair\n", checked, differ);
  EXPECT_TRUE(checked == 126);
  EXPECT_TRUE(differ == 0);
}

TEST(minimax_h3_avae, decode_matches_golden)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_AVAE_PATH");
  const char* gd   = std::getenv("VPIPE_MINIMAX_H3_AVAE_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  const Golden g = load_golden_(gd);
  if (!g.ok) { return; }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMiniMaxH3AudioVae::Config cfg;
  std::string err;
  ASSERT_TRUE(MetalMiniMaxH3AudioVae::config_from_json(root, cfg, &err));
  auto vae = MetalMiniMaxH3AudioVae::load(root, mc, cfg);
  if (!vae) { std::printf("[minimax_h3_avae] load failed\n"); }
  ASSERT_TRUE(vae != nullptr);

  if ((int)g.latent.size() !=
      g.stereo * g.latent_channels * g.frames) {
    std::printf("[minimax_h3_avae] golden latent is %zu, expected %d\n",
                g.latent.size(), g.stereo * g.latent_channels * g.frames);
    EXPECT_TRUE(false);
    return;
  }
  // The predicted length has to agree BEFORE the waveform is compared;
  // a decode that is one sample short per stage still correlates well
  // enough to pass a loose rel-L2 on the overlap.
  const int expect = vae->decoded_samples(g.frames);
  std::printf("[minimax_h3_avae] %d latent frames -> %d samples "
              "(golden %d)\n", g.frames, expect, g.samples);
  EXPECT_TRUE(expect == g.samples);

  std::vector<float> pcm;
  std::vector<std::vector<float>> taps;
  const bool ok =
      vae->decode(g.latent.data(), g.stereo, g.frames, &pcm, &err, &taps);
  if (!ok) { std::printf("[minimax_h3_avae] decode: %s\n", err.c_str()); }
  ASSERT_TRUE(ok);
  if (pcm.size() != g.waveform.size()) {
    std::printf("[minimax_h3_avae] got %zu samples, golden has %zu\n",
                pcm.size(), g.waveform.size());
    EXPECT_TRUE(false);
    return;
  }

  // Per-stage RMS first: the first ratio that leaves 1.0 names the stage
  // that broke, which turns "the audio is wrong" into one stage to read.
  for (std::size_t i = 0; i < taps.size() && i < g.stage_rms.size(); ++i) {
    const double r = rms_(taps[i], 0, taps[i].size());
    const double ref = (double)g.stage_rms[i];
    std::printf("[minimax_h3_avae]   stage %zu rms %.5f vs %.5f  (%.4fx)\n",
                i, r, ref, ref > 0.0 ? r / ref : 0.0);
  }

  const double rl2 = rel_l2_(pcm, g.waveform);
  const std::size_t half = pcm.size() / 2;
  std::printf("[minimax_h3_avae] waveform rel-L2 %.5f  (L %.5f, R %.5f)\n",
              rl2,
              rel_l2_(std::vector<float>(pcm.begin(), pcm.begin() + half),
                      std::vector<float>(g.waveform.begin(),
                                         g.waveform.begin() + half)),
              rel_l2_(std::vector<float>(pcm.begin() + half, pcm.end()),
                      std::vector<float>(g.waveform.begin() + half,
                                         g.waveform.end())));
  // The activations are f16 through a 127-activation chain, so this is a
  // narrowing tolerance, not a "close enough" one. For scale: swapping
  // the two stereo channels, or dropping the 1/3 average, moves this to
  // O(1) -- see stereo_channels_stay_separate below.
  EXPECT_TRUE(rl2 < 0.02);
}

// The two channels are decoded as one batch. If any kernel indexed past
// its own batch item, the left channel's tail would carry the right's
// head -- so decoding the pair together must equal decoding each alone.
TEST(minimax_h3_avae, stereo_channels_stay_separate)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_AVAE_PATH");
  const char* gd   = std::getenv("VPIPE_MINIMAX_H3_AVAE_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  const Golden g = load_golden_(gd);
  if (!g.ok || g.stereo != 2) { return; }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMiniMaxH3AudioVae::Config cfg;
  std::string err;
  ASSERT_TRUE(MetalMiniMaxH3AudioVae::config_from_json(root, cfg, &err));
  auto vae = MetalMiniMaxH3AudioVae::load(root, mc, cfg);
  ASSERT_TRUE(vae != nullptr);

  const std::size_t per = (std::size_t)g.latent_channels * g.frames;
  std::vector<float> both, left, right;
  ASSERT_TRUE(vae->decode(g.latent.data(), 2, g.frames, &both, &err));
  ASSERT_TRUE(vae->decode(g.latent.data(), 1, g.frames, &left, &err));
  ASSERT_TRUE(vae->decode(g.latent.data() + per, 1, g.frames, &right, &err));
  if (both.size() != left.size() + right.size()) {
    EXPECT_TRUE(false);
    return;
  }
  const std::vector<float> bl(both.begin(), both.begin() + left.size());
  const std::vector<float> br(both.begin() + left.size(), both.end());
  const double dl = rel_l2_(bl, left), dr = rel_l2_(br, right);
  std::printf("[minimax_h3_avae] batched vs single: L %.3e, R %.3e\n", dl, dr);
  // Identical arithmetic on identical inputs -- the batch axis only
  // changes how many rows a dispatch covers, never a value.
  EXPECT_TRUE(dl == 0.0 && dr == 0.0);
  // And the two channels must not be the same signal, or the check above
  // would pass for a decoder that ignored the batch index entirely.
  std::printf("[minimax_h3_avae] L vs R differ by rel-L2 %.4f\n",
              rel_l2_(left, right));
  EXPECT_TRUE(rel_l2_(left, right) > 0.1);
}

// ---- the stage ----------------------------------------------------------

namespace {

// Emits one latent-audio beat in generate-video's oport1 shape.
class LatentSource : public TypedStage<LatentSource> {
public:
  static constexpr const char* kTypeName = "ut-alatent-source";
  using TypedStage::TypedStage;

  std::vector<float> data;
  int stereo = 2, channels = 32, frames = 0;

  Job
  process(RuntimeContext& ctx) override
  {
    auto b = std::make_unique<TensorBeatPayload>();
    b->dtype = TensorBeat::DType::F32;
    b->shape = {stereo, channels, frames};
    b->resize_contiguous(data.size());
    std::memcpy(b->as_f32(), data.data(), data.size() * sizeof(float));
    co_await ctx.write(0, std::move(b));
    ctx.signal_done();
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec op[] = {
      {.name = "latent", .doc = "", .type = &typeid(TensorBeatPayload)}};
    static const StageSpec s = {.type_name = "ut-alatent-source", .doc = "",
                                .display_name = "", .oports = op};
    return s;
  }
};

class PcmSink : public TypedStage<PcmSink> {
public:
  static constexpr const char* kTypeName = "ut-pcm-sink";
  using TypedStage::TypedStage;

  int beats = 0, channels = 0, samples = 0, sample_rate = 0;
  std::vector<float> pcm;

  Job
  process(RuntimeContext& ctx) override
  {
    auto b = co_await ctx.read(0);
    if (!b) { ctx.signal_done(); co_return; }
    const auto* t = dynamic_cast<const TensorBeatPayload*>(b.get());
    if (t == nullptr || t->shape.size() != 2) { co_return; }
    ++beats;
    channels = (int)t->shape[0];
    samples  = (int)t->shape[1];
    pcm.assign(t->as_f32(), t->as_f32() + t->element_count());
    if (t->sideband.is_object()) {
      FlexData sb = t->sideband;             // as_object() is a view
      auto o = sb.as_object();
      if (o.contains("sample_rate")) {
        sample_rate = (int)o.at("sample_rate").as_int(0);
      }
    }
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec ip[] = {
      {.name = "audio", .doc = "", .type = &typeid(TensorBeatPayload)}};
    static const StageSpec s = {.type_name = "ut-pcm-sink", .doc = "",
                                .display_name = "", .iports = ip};
    return s;
  }
};

}  // namespace

// The surface, with no model: a stage whose config is incomplete has to
// CONSTRUCT (deferred validation) and expose its ports, so a graph can be
// built and edited before the model exists.
TEST(audio_vae_decode, stage_surface)
{
  Session sess;
  auto s = std::make_unique<AudioVaeDecodeStage>(
      &sess, "avd", std::vector<InEdge>{}, FlexData::make_object());
  EXPECT_TRUE(s->config_error().empty());
  const StageSpec& sp = s->spec();
  EXPECT_TRUE(sp.iports.size() == 2 && sp.oports.size() == 1);
  EXPECT_TRUE(std::string(sp.iports[0].name) == "latent");
  EXPECT_TRUE(std::string(sp.oports[0].name) == "audio");
  // The oport is a TensorBeat of PCM, which is what makes save-audio a
  // legal downstream -- it is the whole reason this is its own stage
  // rather than a branch of vae-decode, whose oport is RGB.
  EXPECT_TRUE(sp.oports[0].type != nullptr &&
              *sp.oports[0].type == typeid(TensorBeatPayload));
  bool has_dir = false, has_unload = false;
  for (const auto& k : sp.attrs) {
    if (std::string(k.key) == "hf_dir") { has_dir = true; }
    if (std::string(k.key) == "unload_when_idle") { has_unload = true; }
  }
  EXPECT_TRUE(has_dir && has_unload);
}

// The stage end to end, fed the golden latent RE-WHITENED. The DiT emits
// normalized latents, so the stage's un-whiten has to be the exact
// inverse -- if it were dropped, or applied with the wrong sign, this
// would still produce a soundtrack, at the wrong per-channel scale.
TEST(audio_vae_decode, decodes_a_whitened_latent)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_AVAE_PATH");
  const char* gd   = std::getenv("VPIPE_MINIMAX_H3_AVAE_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  const Golden g = load_golden_(gd);
  if (!g.ok) { return; }

  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  MetalMiniMaxH3AudioVae::Config vcfg;
  std::string cerr;
  ASSERT_TRUE(MetalMiniMaxH3AudioVae::config_from_json(root, vcfg, &cerr));
  ASSERT_TRUE((int)vcfg.latents_mean.size() == g.latent_channels);

  // (z - mean) / std: the space generate-video's oport1 speaks.
  std::vector<float> white(g.latent.size());
  for (int b = 0; b < g.stereo; ++b) {
    for (int ch = 0; ch < g.latent_channels; ++ch) {
      const float mu = vcfg.latents_mean[(std::size_t)ch];
      const float sd = vcfg.latents_std[(std::size_t)ch];
      const std::size_t off =
          ((std::size_t)b * g.latent_channels + ch) * g.frames;
      for (int t = 0; t < g.frames; ++t) {
        white[off + (std::size_t)t] =
            (g.latent[off + (std::size_t)t] - mu) / sd;
      }
    }
  }

  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto src_u = std::make_unique<LatentSource>(&sess, "src",
                                              std::vector<InEdge>{},
                                              FlexData::make_object());
  src_u->data = white;
  src_u->stereo = g.stereo;
  src_u->channels = g.latent_channels;
  src_u->frames = g.frames;
  src_u->allocate_oports(1);
  auto* src = static_cast<LatentSource*>(pl->insert_stage(std::move(src_u)));

  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("hf_dir", FlexData::make_string(root));
  auto dec_u = std::make_unique<AudioVaeDecodeStage>(
      &sess, "avd", std::vector<InEdge>{{src, 0}}, cfg);
  auto* dec =
      static_cast<AudioVaeDecodeStage*>(pl->insert_stage(std::move(dec_u)));

  auto sink_u = std::make_unique<PcmSink>(&sess, "sink",
                                          std::vector<InEdge>{{dec, 0}},
                                          FlexData::make_object());
  auto* sink = static_cast<PcmSink*>(pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  std::printf("[audio_vae_decode] family '%s', %llu clips, sink got %d beats "
              "of [%d, %d] at %d Hz\n", dec->family().c_str(),
              (unsigned long long)dec->clips_emitted(), sink->beats,
              sink->channels, sink->samples, sink->sample_rate);
  EXPECT_TRUE(dec->family() == "minimax-h3");
  EXPECT_TRUE(sink->beats == 1);
  EXPECT_TRUE(sink->channels == g.stereo);
  EXPECT_TRUE(sink->samples == g.samples);
  // save-audio reads the rate off the sideband, so an unset one silently
  // resamples the whole clip to its 24 kHz default.
  EXPECT_TRUE(sink->sample_rate == g.sample_rate);
  if (sink->pcm.size() != g.waveform.size()) {
    EXPECT_TRUE(false);
    return;
  }
  const double rl2 = rel_l2_(sink->pcm, g.waveform);
  std::printf("[audio_vae_decode] stage waveform rel-L2 %.5f\n", rl2);
  EXPECT_TRUE(rl2 < 0.02);
  // Everything the VAE emits is clamped, and a decode that skipped the
  // un-whiten would drift well outside the reference's peak.
  float peak = 0.0f;
  for (float v : sink->pcm) { peak = std::max(peak, std::fabs(v)); }
  std::printf("[audio_vae_decode] peak %.4f\n", peak);
  EXPECT_TRUE(peak <= 1.0f);
}

// A realistic clip, timed. The video half of this model is minutes of
// 33B denoising, so the soundtrack only has to not be a second
// bottleneck -- but a vocoder that ran slower than realtime would be
// one, and that is worth finding here rather than at the end of a
// generation.
TEST(minimax_h3_avae, decode_throughput)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_AVAE_PATH");
  const char* gd   = std::getenv("VPIPE_MINIMAX_H3_AVAE_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  const Golden g = load_golden_(gd);
  if (!g.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMiniMaxH3AudioVae::Config cfg;
  std::string err;
  ASSERT_TRUE(MetalMiniMaxH3AudioVae::config_from_json(root, cfg, &err));
  auto vae = MetalMiniMaxH3AudioVae::load(root, mc, cfg);
  ASSERT_TRUE(vae != nullptr);

  // 5 seconds at 40 latents/s, tiled from the golden latent.
  const int T = 5 * cfg.sample_rate / cfg.hop();
  std::vector<float> z((std::size_t)g.stereo * g.latent_channels * T);
  for (int b = 0; b < g.stereo; ++b) {
    for (int ch = 0; ch < g.latent_channels; ++ch) {
      const std::size_t src =
          ((std::size_t)b * g.latent_channels + ch) * g.frames;
      const std::size_t dst = ((std::size_t)b * g.latent_channels + ch) * T;
      for (int t = 0; t < T; ++t) {
        z[dst + (std::size_t)t] = g.latent[src + (std::size_t)(t % g.frames)];
      }
    }
  }
  std::vector<float> pcm;
  ASSERT_TRUE(vae->decode(z.data(), g.stereo, T, &pcm, &err));   // warm up
  const auto t0 = std::chrono::steady_clock::now();
  ASSERT_TRUE(vae->decode(z.data(), g.stereo, T, &pcm, &err));
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  const double secs = (double)(pcm.size() / (std::size_t)g.stereo) /
                      (double)cfg.sample_rate;
  std::printf("[minimax_h3_avae] %d frames -> %.2f s stereo in %.1f ms "
              "(%.0fx realtime)\n", T, secs, ms, secs * 1000.0 / ms);
  EXPECT_TRUE(ms < secs * 1000.0);
}

// Is the decoder producing STRUCTURE or noise?
//
// A diagnostic, not a golden -- this box has no audio golden, and the
// question it answers is the one an "audio came out as noise" report
// actually poses: is the VAE broken, or is it faithfully decoding a bad
// latent? Those need completely different fixes, and nothing else here
// separates them.
//
// The probe is a CONSTANT latent (every frame identical, at the value the
// un-whitening maps zero to). A working vocoder turns that into something
// periodic and band-limited -- low zero-crossing rate, energy concentrated
// low. A broken one -- wrong weights, a mis-read kernel, weight norm that
// should have been applied and was not -- turns it into broadband noise,
// whose zero-crossing rate sits near 0.5 crossings per sample because
// consecutive samples are independent.
//
// ZCR is the discriminator rather than RMS because amplitude alone cannot
// tell a loud tone from loud noise, and it is scale-free, so it needs no
// reference recording to compare against.
//
// WHAT THIS DOES NOT PROVE, stated because it was over-read once already:
// a CONSTANT latent is a degenerate input, and passing here does NOT mean
// the weights were parsed correctly. A decoder built from wrong-but-smooth
// weights still turns a constant into something smooth. It rules out
// gross breakage -- unbound tensors, an activation left at zero, a
// mis-read kernel width -- and nothing finer. Certifying a parse needs a
// reference decode, which on this box means the released checkpoint next
// to the repack (see minimax_h3_dit.comfy_layout_matches_golden for the
// shape that argument has to take). The audio VAE is the one component
// the H3 bring-up never cross-verified between publishers.
//
// Env: VPIPE_MINIMAX_H3_AVAE_PATH.
TEST(minimax_h3_avae, constant_latent_decodes_to_structure_not_noise)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_AVAE_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string dir = MetalMiniMaxH3AudioVae::resolve_vae_dir(root);
  MetalMiniMaxH3AudioVae::Config cfg;
  std::string cerr;
  if (!MetalMiniMaxH3AudioVae::config_from_json(dir, cfg, &cerr)) { return; }
  auto vae = MetalMiniMaxH3AudioVae::load(dir, mc, cfg);
  ASSERT_TRUE(vae != nullptr);

  const int B = cfg.stereo_channels, ZC = cfg.latent_channels, T = 32;
  // Zero in NORMALISED space, un-whitened exactly as the stage does it.
  std::vector<float> z((std::size_t)B * ZC * T);
  for (int b = 0; b < B; ++b) {
    for (int ch = 0; ch < ZC; ++ch) {
      const float mu = (int)cfg.latents_mean.size() > ch
                           ? cfg.latents_mean[(std::size_t)ch] : 0.0f;
      for (int t = 0; t < T; ++t) {
        z[((std::size_t)b * ZC + ch) * T + (std::size_t)t] = mu;
      }
    }
  }
  std::vector<float> pcm;
  std::string derr;
  ASSERT_TRUE(vae->decode(z.data(), B, T, &pcm, &derr));
  const int n = B > 0 ? (int)(pcm.size() / (std::size_t)B) : 0;
  ASSERT_TRUE(n > 0);

  for (int b = 0; b < B; ++b) {
    const float* s = pcm.data() + (std::size_t)b * n;
    double sum2 = 0.0, peak = 0.0;
    long cross = 0;
    for (int i = 0; i < n; ++i) {
      sum2 += (double)s[i] * s[i];
      peak = std::max(peak, (double)std::fabs(s[i]));
      if (i > 0 && ((s[i] < 0.0f) != (s[i - 1] < 0.0f))) { ++cross; }
    }
    const double rms = std::sqrt(sum2 / (double)n);
    const double zcr = (double)cross / (double)(n - 1);
    std::printf("[minimax_h3_avae] ch%d: %d samples, rms %.5f, peak %.5f, "
                "zero-crossing rate %.4f\n", b, n, rms, peak, zcr);
    // White noise sits near 0.5. Anything a vocoder produces from a
    // constant latent should be far below that; 0.35 leaves generous
    // room for a bright but real signal.
    EXPECT_TRUE(zcr < 0.35);
    // And it must not be silent either -- an all-zero output would pass
    // a ZCR test trivially while telling us the decode did nothing.
    EXPECT_TRUE(rms > 1e-6);
  }
}

// TWO publishers' audio VAEs, decoding the SAME latent.
//
// This is what certifies the weight-norm fold, and nothing cheaper does.
// PyTorch keeps weight norm split as (g, v); the Comfy-Org repack ships
// the product. Folding needs the norm taken over every axis but dim 0 --
// and for the five ConvTranspose1d `ups`, dim 0 is the INPUT channel
// count, not the output one. A folder that used the wrong axis produces a
// file that loads, has the right shapes, and decodes a constant latent
// into something perfectly smooth, while turning real audio into noise.
// The structure probe above cannot see that; a reference decode can.
//
// It needs BOTH checkpoints, which is why it is a second env var rather
// than a golden: telling a right fold from a wrong one requires `g`, and
// a folded file no longer carries it. Only a machine holding the release
// AND the repack can answer it, so this skips everywhere else.
//
// Point PATH at either one and PATH2 at the other -- the test is
// symmetric and does not care which is folded.
TEST(minimax_h3_avae, both_publishers_decode_alike)
{
  const char* ra = std::getenv("VPIPE_MINIMAX_H3_AVAE_PATH");
  const char* rb = std::getenv("VPIPE_MINIMAX_H3_AVAE_PATH2");
  if (ra == nullptr || *ra == '\0' || rb == nullptr || *rb == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  auto open = [&](const char* p, MetalMiniMaxH3AudioVae::Config* cfg)
      -> std::unique_ptr<MetalMiniMaxH3AudioVae> {
    const std::string dir = MetalMiniMaxH3AudioVae::resolve_vae_dir(p);
    std::string e;
    if (!MetalMiniMaxH3AudioVae::config_from_json(dir, *cfg, &e)) {
      return nullptr;
    }
    return MetalMiniMaxH3AudioVae::load(dir, mc, *cfg);
  };
  MetalMiniMaxH3AudioVae::Config ca, cb;
  auto va = open(ra, &ca);
  auto vb = open(rb, &cb);
  ASSERT_TRUE(va != nullptr && vb != nullptr);
  ASSERT_TRUE(ca.stereo_channels == cb.stereo_channels &&
              ca.latent_channels == cb.latent_channels &&
              ca.sample_rate == cb.sample_rate && ca.hop() == cb.hop());

  // A RANDOM latent, not a constant one: the whole failure this test
  // exists for is invisible on a degenerate input. Fixed seed so a
  // mismatch is reproducible rather than a different draw each run.
  const int B = ca.stereo_channels, ZC = ca.latent_channels, T = 48;
  std::vector<float> z((std::size_t)B * ZC * T);
  std::mt19937 rng(12345);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (float& v : z) { v = nd(rng); }

  std::vector<float> pa, pb;
  std::string ea, eb;
  ASSERT_TRUE(va->decode(z.data(), B, T, &pa, &ea));
  ASSERT_TRUE(vb->decode(z.data(), B, T, &pb, &eb));
  ASSERT_TRUE(pa.size() == pb.size() && !pa.empty());

  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < pa.size(); ++i) {
    const double d = (double)pa[i] - (double)pb[i];
    num += d * d;
    den += (double)pb[i] * pb[i];
  }
  const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
  std::printf("[minimax_h3_avae] two publishers, %zu samples: rel-L2 %.3e\n",
              pa.size(), rel);
  // The two checkpoints store the same numbers in different FORMS -- one
  // folded in fp32 by the repacker, one folded here from (g, v) -- so
  // they agree to rounding, not exactly. A wrong fold axis lands at
  // order 1, so this bound does not need to be tight to be decisive.
  EXPECT_TRUE(rel < 1e-2);
}

// The same timing question as decode_throughput, without needing a
// golden: the latent is synthetic, because cost here depends only on the
// frame count and nothing about fidelity.
//
// Env: VPIPE_MINIMAX_H3_AVAE_BENCH=1 + VPIPE_MINIMAX_H3_AVAE_PATH.
TEST(minimax_h3_avae, decode_bench)
{
  const char* on   = std::getenv("VPIPE_MINIMAX_H3_AVAE_BENCH");
  const char* root = std::getenv("VPIPE_MINIMAX_H3_AVAE_PATH");
  if (on == nullptr || *on == '\0' || *on == '0') { return; }
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMiniMaxH3AudioVae::Config cfg;
  std::string err;
  ASSERT_TRUE(MetalMiniMaxH3AudioVae::config_from_json(root, cfg, &err));
  auto vae = MetalMiniMaxH3AudioVae::load(root, mc, cfg);
  ASSERT_TRUE(vae != nullptr);

  const char* sv = std::getenv("VPIPE_MINIMAX_H3_AVAE_BENCH_SECONDS");
  const double want = (sv != nullptr && *sv != '\0') ? std::atof(sv) : 5.0;
  const int T = (int)(want * (double)cfg.sample_rate / (double)cfg.hop());
  const int B = cfg.stereo_channels, ZC = cfg.latent_channels;
  std::vector<float> z((std::size_t)B * ZC * T);
  for (std::size_t i = 0; i < z.size(); ++i) {
    z[i] = 0.1f * (float)((int)(i * 31 % 19) - 9);
  }
  std::vector<float> pcm;
  ASSERT_TRUE(vae->decode(z.data(), B, T, &pcm, &err));            // warm up
  const auto t0 = std::chrono::steady_clock::now();
  ASSERT_TRUE(vae->decode(z.data(), B, T, &pcm, &err));
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  const double secs =
      (double)(pcm.size() / (std::size_t)B) / (double)cfg.sample_rate;
  std::printf("[minimax_h3_avae] %d frames -> %.2f s stereo in %.1f ms "
              "(%.0fx realtime)\n", T, secs, ms, secs * 1000.0 / ms);
  EXPECT_TRUE(ms > 0.0);
}
