#include "stages/generate-video-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/wan/metal-wan-vae.h"
#include "generative-models/weight-set.h"
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "Wan model root (transformer/, transformer_2/, vae/, text_encoder/). "
          "OPTIONAL: a model-select source on the model iport overrides it",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "wan-i2v,wan-t2v"},
  {.key = "height", .type = ConfigType::Int, .required = false,
   .doc = "video height in pixels; a multiple of 16 (the VAE's 8x times the "
          "DiT's 2x patch)", .def_int = 480},
  {.key = "width", .type = ConfigType::Int, .required = false,
   .doc = "video width in pixels; a multiple of 16", .def_int = 832},
  {.key = "frames", .type = ConfigType::Int, .required = false,
   .doc = "video frames. F % 4 == 1 (81, 121, ...) because the VAE's temporal "
          "chunking makes the first frame its own chunk", .def_int = 81},
  {.key = "fps", .type = ConfigType::Real, .required = false,
   .doc = "frame rate stamped on the emitted latent, for the decoder and the "
          "encoder downstream of it", .def_real = 16.0},
  {.key = "steps", .type = ConfigType::Int, .required = false,
   .doc = "denoising steps", .def_int = 40},
  {.key = "seed", .type = ConfigType::Int, .required = false,
   .doc = "initial-noise RNG seed (default 0)"},
  {.key = "guidance_scale", .type = ConfigType::Real, .required = false,
   .doc = "classifier-free guidance for the HIGH-noise expert. Needs a "
          "negative conditioning on iport1; without one, guidance is forced "
          "to 1", .def_real = 3.5},
  {.key = "guidance_scale_2", .type = ConfigType::Real, .required = false,
   .doc = "classifier-free guidance for the LOW-noise expert (A14B only)",
   .def_real = 3.5},
  {.key = "boundary_ratio", .type = ConfigType::Real, .required = false,
   .doc = "sigma at which the low-noise expert takes over from the high-noise "
          "one. Read from the checkpoint's model_index.json when present; 0 "
          "means a single expert", .def_real = 0.9},
  {.key = "unload_when_idle", .type = ConfigType::String, .required = false,
   .doc = "drop the expert's weights after each clip and reload on the next "
          "one. \"auto\" (default) decides from physical RAM vs the "
          "pipeline's weight bytes; \"always\" / \"never\" force it",
   .def_str = "auto"},
};
const PortSpec kIports[] = {
  {.name = "conditioning",
   .doc = "umT5-XXL hidden states from a diffusion-conditioner: bf16 "
          "[text_seq, 4096]",
   .type = &typeid(TensorBeatPayload),
   .tags = "conditioning", .clock_group = 0},
  {.name = "neg_conditioning",
   .doc = "OPTIONAL negative conditioning (the conditioner's oport1) for "
          "classifier-free guidance. Wan is not distilled, so without this "
          "guidance is forced to 1",
   .type = &typeid(TensorBeatPayload),
   .tags = "conditioning", .clock_group = 0},
  {.name = "model", .doc = "OPTIONAL shared model reference from a model-select "
                           "source; overrides the hf_dir config",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "sampler",
   .doc = "OPTIONAL sampler spec FlexData (diffusion-sampler-select); the "
          "default is UniPC multistep, which is what Wan ships",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "scheduler",
   .doc = "OPTIONAL scheduler spec FlexData (scheduler-select)",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "ref_latent0",
   .doc = "OPTIONAL image-to-video conditioning latent from vae-encode: f32 "
          "[16, T, H/8, W/8], the VAE encoding of the conditioning image "
          "followed by blank frames. Present => image-to-video",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};
[[maybe_unused]] constexpr unsigned kModelPort   = 2;
[[maybe_unused]] constexpr unsigned kSamplerPort = 3;
[[maybe_unused]] constexpr unsigned kSchedPort   = 4;
[[maybe_unused]] constexpr unsigned kRefPort     = 5;

const PortSpec kOports[] = {
  {.name = "latent",
   .doc = "f32 latent VIDEO [16, T, H/8, W/8] (whitened), sideband "
          "{fps, frames}",
   .type = &typeid(TensorBeatPayload),
   .tags = "latent", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "generate-video",
  .doc       = "Wan video DiT denoiser: conditioning (+ an optional image "
               "latent) -> UniPC over a two-expert 14B transformer -> a latent "
               "video, on the metal-compute backend. Feed vae-decode.",
  .display_name = "Generate Video",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

#ifdef VPIPE_BUILD_APPLE_SILICON
// boundary_ratio lives in the PIPELINE config (model_index.json), not in
// either expert's own config, because it is a property of the pair.
double
boundary_from_index_(const std::string& root, double fallback)
{
  std::ifstream in(std::filesystem::path(root) / "model_index.json");
  if (!in) { return fallback; }
  FlexData fd;
  try {
    fd = FlexData::from_json(in);
  } catch (...) {
    return fallback;
  }
  if (!fd.is_object()) { return fallback; }
  auto o = fd.as_object();
  if (!o.contains("boundary_ratio")) { return fallback; }
  FlexData br = o.at("boundary_ratio");
  if (br.is_null()) { return 0.0; }   // explicitly a single-expert pipeline
  return br.as_real(fallback);
}
#endif

}  // namespace

GenerateVideoStage::GenerateVideoStage(const SessionContextIntf* s,
                                       std::string               id,
                                       std::vector<InEdge>       iports,
                                       FlexData                  config)
  : TypedStage<GenerateVideoStage>(s, std::move(id), std::move(iports),
                                   std::move(config))
{
  _hf_dir = attr_str("hf_dir");
  _height = (int)attr_int("height");
  _width  = (int)attr_int("width");
  _frames = (int)attr_int("frames");
  _fps    = attr_real("fps");
  _steps  = (int)attr_int("steps");
  _seed   = (std::uint64_t)attr_int("seed");
  _guidance   = attr_real("guidance_scale");
  _guidance_2 = attr_real("guidance_scale_2");
  _boundary   = attr_real("boundary_ratio");
  if (_height <= 0) { _height = 480; }
  if (_width  <= 0) { _width  = 832; }
  if (_frames <= 0) { _frames = 81; }
  if (_steps  <= 0) { _steps  = 40; }
  if (!(_fps > 0.0)) { _fps = 16.0; }
  // Deferred validation: never throw from a constructor, so a geometry the
  // model cannot represent is reported and the runtime skips the stage.
  if ((_height % 16) != 0 || (_width % 16) != 0) {
    fail_config(fmt("height {} and width {} must both be multiples of 16 "
                    "(the VAE's 8x downsample times the DiT's 2x patch)",
                    _height, _width));
  }
  // F % 4 == 1 is not a rounding convenience: the VAE's first chunk is a
  // single frame and every later one is four, so any other count has no
  // latent representation at all.
  if ((_frames % 4) != 1) {
    fail_config(fmt("frames {} is not 4k+1; the video VAE compresses time in "
                    "4-frame chunks after a 1-frame first chunk, so use 81, "
                    "121, ...", _frames));
  }
#ifdef VPIPE_BUILD_APPLE_SILICON
  {
    bool bad = false;
    _unload_cfg = model_memory::parse_unload_policy(
        attr_str("unload_when_idle"), &bad);
    if (bad) {
      session()->warn(fmt(
          "GenerateVideoStage('{}'): unload_when_idle '{}' is not "
          "auto|always|never; using auto", this->id(),
          attr_str("unload_when_idle")));
    }
  }
  // UniPC multistep on the checkpoint's flow schedule -- the sampler every
  // Wan checkpoint ships with. A sampler-select source overrides it.
  _sampler_spec.method = "unipc";
  _sampler_spec.order = 2;
  _sampler_spec.solver_bh2 = true;
  _scheduler_spec.type = "unipc_flow";
  _scheduler_spec.steps = _steps;
  _scheduler_spec.shift = 3.0;
  _scheduler_spec.num_train = 1000;
#endif
  allocate_oports(spec().oports.size());
}

GenerateVideoStage::~GenerateVideoStage() = default;

const StageSpec&
GenerateVideoStage::spec() const noexcept
{
  return kSpec;
}

int
GenerateVideoStage::latent_frames() const noexcept
{
  return _frames > 0 ? 1 + (_frames - 1) / 4 : 0;
}

std::vector<ResourceClaim>
GenerateVideoStage::declare_resources() const
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  if (_hf_dir.empty()) { return {}; }
  namespace fs = std::filesystem;
  const std::string root = resolve_model_dir(session(), _hf_dir);
  // ONE expert, not both. The stage holds exactly one at a time, so
  // claiming the pair would size every peer against a peak that never
  // occurs and push them all into streaming for nothing.
  return model_memory::weight_claims({(fs::path(root) / "transformer").string()});
#else
  return {};
#endif
}

void
GenerateVideoStage::reset_run_state()
{
  _emitted = 0;
  _model_latched = false;
  _sampler_latched = false;
  _scheduler_latched = false;
}

#ifdef VPIPE_BUILD_APPLE_SILICON

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

void
GenerateVideoStage::resolve_config_()
{
  if (_have_cfg || _hf_dir.empty()) { return; }
  namespace fs = std::filesystem;
  _root = resolve_model_dir(session(), _hf_dir);
  const std::string t1 = (fs::path(_root) / "transformer").string();
  std::string cerr;
  if (!genai::MetalWanTransformer::config_from_json(t1, _cfg, &cerr)) {
    session()->error(fmt(
        "GenerateVideoStage('{}'): {}; inert", this->id(), cerr));
    return;
  }
  _two_experts = fs::exists(fs::path(_root) / "transformer_2" / "config.json");
  _boundary = boundary_from_index_(_root, _boundary);
  if (!_two_experts) { _boundary = 0.0; }
  _have_cfg = true;
  if (!_unload_resolved) {
    _unload_resolved = true;
    const std::vector<std::string> peers = {
        (fs::path(_root) / "text_encoder").string(),
        (fs::path(_root) / "vae").string()};
    switch (_unload_cfg) {
      case model_memory::UnloadPolicy::kAlways: _unload_idle = true;  break;
      case model_memory::UnloadPolicy::kNever:  _unload_idle = false; break;
      default:
        _unload_idle = model_memory::bounded(session(), peers,
                                             model_memory::kHeadroom);
        break;
    }
  }
  session()->info(fmt(
      "GenerateVideoStage('{}'): Wan {} at {}x{}x{} frames, {} steps, "
      "in_channels {}{}", this->id(),
      _two_experts ? "A14B (two experts)" : "single-expert",
      _width, _height, _frames, _steps, _cfg.in_channels,
      _two_experts ? fmt(", boundary {:.2f}", _boundary)() : std::string()));
}

bool
GenerateVideoStage::ensure_expert_(int which)
{
  if (_dit && _expert == which) { return true; }
  resolve_config_();
  if (!_have_cfg) { return false; }
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr) { return false; }
  namespace fs = std::filesystem;
  const std::string dir =
      (fs::path(_root) / (which == 0 ? "transformer" : "transformer_2"))
          .string();
  // Drop first, THEN load. At bf16 an expert is ~28 GB, so holding both
  // across the swap is not a transient peak, it is an out-of-memory kill.
  if (_dit) {
    _dit.reset();
    _expert = -1;
  }
  session()->info(fmt(
      "GenerateVideoStage('{}'): loading the {}-noise expert from '{}'",
      this->id(), which == 0 ? "high" : "low", dir));
  std::shared_ptr<genai::WeightSet> ws = genai::open_weight_set(dir, session());
  if (!ws) {
    session()->error(fmt(
        "GenerateVideoStage('{}'): no readable checkpoint under '{}'",
        this->id(), dir));
    return false;
  }
  _dit = genai::MetalWanTransformer::load(ws, mc, _cfg);
  if (!_dit) {
    session()->error(fmt(
        "GenerateVideoStage('{}'): failed to load the DiT from '{}'",
        this->id(), dir));
    return false;
  }
  _expert = which;
  return true;
}

Job
GenerateVideoStage::initialize(RuntimeContext& ctx)
{
  // Defer everything when a model-select source feeds the model iport (its
  // beat only arrives after the init barrier). The expert itself is loaded
  // lazily at the first denoise step regardless -- it is 28 GB, and a graph
  // that never receives a prompt should not pay for it.
  if (!(ctx.num_iports() > kModelPort && ctx.iport_connected(kModelPort))) {
    resolve_config_();
  }
  co_return;
}

Job
GenerateVideoStage::process(RuntimeContext& ctx)
{
  if (!_model_latched && ctx.num_iports() > kModelPort &&
      ctx.iport_connected(kModelPort)) {
    auto mb = co_await ctx.read(kModelPort);
    _model_latched = true;
    if (const auto* mfd =
            mb ? dynamic_cast<const FlexDataPayload*>(mb.get()) : nullptr) {
      if (apply_model_select_beat(mfd->data, _hf_dir)) { resolve_config_(); }
    }
  }
  if (!_sampler_latched && ctx.num_iports() > kSamplerPort &&
      ctx.iport_connected(kSamplerPort)) {
    auto sb = co_await ctx.read(kSamplerPort);
    _sampler_latched = true;
    if (const auto* sfd =
            sb ? dynamic_cast<const FlexDataPayload*>(sb.get()) : nullptr) {
      std::string serr;
      _sampler_spec = genai::FlowSamplerSpec::from_flex(sfd->data, &serr);
      if (!serr.empty()) {
        session()->warn(fmt("GenerateVideoStage('{}'): sampler spec: {}",
                            this->id(), serr));
      }
    }
  }
  if (!_scheduler_latched && ctx.num_iports() > kSchedPort &&
      ctx.iport_connected(kSchedPort)) {
    auto cb = co_await ctx.read(kSchedPort);
    _scheduler_latched = true;
    if (const auto* cfd =
            cb ? dynamic_cast<const FlexDataPayload*>(cb.get()) : nullptr) {
      std::string cerr;
      _scheduler_spec = genai::FlowSchedulerSpec::from_flex(cfd->data, &cerr);
      if (!cerr.empty()) {
        session()->warn(fmt("GenerateVideoStage('{}'): scheduler spec: {}",
                            this->id(), cerr));
      }
    }
  }

  auto in = co_await ctx.read(0);
  if (!in) { ctx.signal_done(); co_return; }
  const auto* cond = dynamic_cast<const TensorBeatPayload*>(in.get());
  if (cond == nullptr || cond->shape.size() != 2) {
    session()->warn(fmt(
        "GenerateVideoStage('{}'): expected a [text_seq, {}] conditioning "
        "TensorBeat, got {}; skipping", this->id(), _cfg.text_dim,
        in->describe()));
    co_return;
  }
  // The negative is emitted BEFORE the positive by the conditioner, so by
  // the time the positive lands its pair is already queued -- a non-blocking
  // poll rather than a read that could deadlock when there is no negative.
  std::unique_ptr<BeatPayloadIntf> negb;
  if (ctx.num_iports() > 1 && ctx.iport_connected(1) && ctx.backlog(1) > 0) {
    negb = co_await ctx.read(1);
  }
  const auto* neg =
      negb ? dynamic_cast<const TensorBeatPayload*>(negb.get()) : nullptr;
  std::unique_ptr<BeatPayloadIntf> refb;
  if (ctx.num_iports() > kRefPort && ctx.iport_connected(kRefPort)) {
    refb = co_await ctx.read(kRefPort);
  }
  const auto* ref =
      refb ? dynamic_cast<const TensorBeatPayload*>(refb.get()) : nullptr;

  if (!ensure_expert_(0)) {
    session()->warn(fmt(
        "GenerateVideoStage('{}'): no DiT; skipping", this->id()));
    co_return;
  }
  auto* mc = session()->services()->metal_compute();
  const int ZC = 16;                       // the VAE's latent channels
  const int T  = latent_frames();
  const int h8 = _height / 8, w8 = _width / 8;
  const int text_seq = (int)cond->shape[0];
  const std::size_t nlat = (std::size_t)ZC * T * h8 * w8;
  const std::size_t plane = (std::size_t)h8 * w8;

  // ---- image-to-video conditioning channels -------------------------
  // in_channels 36 = 16 noise + 4 first-frame mask + 16 image latent. The
  // mask is 4 channels rather than 1 because the VAE's 4x temporal
  // compression folds four video frames into one latent frame: latent
  // frame 0 covers video frame 0 alone (the conditioning image) plus three
  // repeats of it, and every later latent frame covers four blank ones.
  const bool i2v = _cfg.in_channels > ZC;
  std::vector<float> cond_ch;              // [20, T, h8, w8] = mask + image
  if (i2v) {
    if (ref == nullptr || ref->shape.size() != 4 ||
        (int)ref->shape[0] != ZC || (int)ref->shape[1] != T ||
        (int)ref->shape[2] != h8 || (int)ref->shape[3] != w8) {
      session()->warn(fmt(
          "GenerateVideoStage('{}'): this checkpoint has in_channels {}, so it "
          "is image-to-video and needs a [{}, {}, {}, {}] conditioning latent "
          "on ref_latent0 (from a vae-encode over the conditioning image); "
          "skipping", this->id(), _cfg.in_channels, ZC, T, h8, w8));
      co_return;
    }
    cond_ch.assign((std::size_t)20 * T * plane, 0.0f);
    // Mask channel c of latent frame t is 1 exactly where the video frame
    // it stands for is the conditioning image: (t=0, c=0..3) covers video
    // frame 0 four times over, and nothing else does.
    for (int c = 0; c < 4; ++c) {
      float* m = cond_ch.data() + ((std::size_t)c * T) * plane;
      for (std::size_t i = 0; i < plane; ++i) { m[i] = 1.0f; }
    }
    std::memcpy(cond_ch.data() + (std::size_t)4 * T * plane, ref->as_f32(),
                nlat * sizeof(float));
  }

  // ---- schedule + initial noise --------------------------------------
  genai::FlowSchedulerSpec sched = _scheduler_spec;
  if (sched.steps <= 0) { sched.steps = _steps; }
  genai::FlowSampler sampler(_sampler_spec, sched);
  sampler.reset();
  const std::vector<double>& sigmas = sampler.sigmas();

  std::vector<float> x(nlat);
  {
    std::mt19937_64 rng(_seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x) { v = nd(rng); }
  }

  // Text conditioning is projected by the EXPERT's own text_embedder, so it
  // is re-projected on a swap rather than computed once for the clip.
  SharedBuffer tpos, tneg;
  int expert_for_text = -1;
  auto upload_bf16 = [&](const TensorBeatPayload& b) {
    const std::size_t n = (std::size_t)b.shape[0] * b.shape[1];
    SharedBuffer s = mc->make_shared_buffer(n * 2);
    if (!s.empty()) { std::memcpy(s.contents(), b.as_bf16(), n * 2); }
    return s;
  };
  const SharedBuffer cond_raw = upload_bf16(*cond);
  const SharedBuffer neg_raw = neg != nullptr ? upload_bf16(*neg)
                                              : SharedBuffer{};
  const bool cfg_on = neg != nullptr;

  // The DiT input buffer, rebuilt per evaluation from the sampler's
  // candidate latent plus the constant conditioning channels.
  SharedBuffer dit_in =
      mc->make_shared_buffer((std::size_t)_cfg.in_channels * T * plane * 2);
  if (dit_in.empty()) {
    session()->warn(fmt("GenerateVideoStage('{}'): latent allocation failed",
                        this->id()));
    co_return;
  }

  bool failed = false;
  auto denoise = [&](const std::vector<float>& xin,
                     double sigma) -> std::vector<float> {
    std::vector<float> out(nlat, 0.0f);
    if (failed) { return out; }
    // Two experts, one boundary, one crossing: the schedule descends, so
    // this swaps at most once per clip.
    const int want = (_boundary > 0.0 && sigma < _boundary) ? 1 : 0;
    if (!ensure_expert_(want)) { failed = true; return out; }
    if (_expert != expert_for_text) {
      std::string terr;
      tpos = _dit->encode_text(cond_raw, text_seq, &terr);
      tneg = cfg_on ? _dit->encode_text(neg_raw, (int)neg->shape[0], &terr)
                    : SharedBuffer{};
      if (tpos.empty()) {
        session()->warn(fmt("GenerateVideoStage('{}'): text projection: {}",
                            this->id(), terr));
        failed = true;
        return out;
      }
      expert_for_text = _expert;
    }
    auto* dp = static_cast<std::uint16_t*>(dit_in.contents());
    auto put = [&](std::size_t i, float v) {
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      dp[i] = (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    };
    for (std::size_t i = 0; i < nlat; ++i) { put(i, xin[i]); }
    for (std::size_t i = 0; i < cond_ch.size(); ++i) {
      put(nlat + i, cond_ch[i]);
    }
    // The scheduler's timestep is sigma on its native 0..num_train scale,
    // which is what the DiT's sinusoidal embedding was trained against.
    const float ts = (float)(sigma * (double)sched.num_train);
    const double g = (_expert == 1) ? _guidance_2 : _guidance;
    std::string derr;
    SharedBuffer vp =
        _dit->forward(dit_in, T, h8, w8, tpos, text_seq, ts, &derr);
    if (vp.empty()) {
      session()->warn(fmt("GenerateVideoStage('{}'): DiT forward: {}",
                          this->id(), derr));
      failed = true;
      return out;
    }
    auto bf16 = [](std::uint16_t b) {
      const std::uint32_t u = (std::uint32_t)b << 16;
      float f;
      std::memcpy(&f, &u, 4);
      return f;
    };
    const auto* pp = static_cast<const std::uint16_t*>(vp.contents());
    if (cfg_on && g != 1.0) {
      SharedBuffer vn = _dit->forward(dit_in, T, h8, w8, tneg,
                                      (int)neg->shape[0], ts, &derr);
      if (vn.empty()) {
        session()->warn(fmt("GenerateVideoStage('{}'): negative forward: {}",
                            this->id(), derr));
        failed = true;
        return out;
      }
      const auto* np = static_cast<const std::uint16_t*>(vn.contents());
      for (std::size_t i = 0; i < nlat; ++i) {
        const float u = bf16(np[i]);
        out[i] = u + (float)g * (bf16(pp[i]) - u);
      }
    } else {
      for (std::size_t i = 0; i < nlat; ++i) { out[i] = bf16(pp[i]); }
    }
    return out;
  };

  {
    PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmDit, kPerfLlmDitBegin,
                       (std::uint64_t)T * (h8 / 2) * (w8 / 2));
    for (int i = 0; i < sched.steps && !failed; ++i) {
      session()->log_debug(fmt(
          "GenerateVideoStage('{}'): step {}/{} sigma {:.4f} ({}-noise "
          "expert)", this->id(), i + 1, sched.steps, sigmas[(std::size_t)i],
          (_boundary > 0.0 && sigmas[(std::size_t)i] < _boundary) ? "low"
                                                                  : "high"));
      sampler.step(i, x, denoise);
    }
  }
  if (failed) { co_return; }

  if (_unload_idle) {
    _dit.reset();
    _expert = -1;
    expert_for_text = -1;
  }

  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::F32;
  out->shape = {ZC, T, h8, w8};
  out->resize_contiguous(nlat);
  std::memcpy(out->as_f32(), x.data(), nlat * sizeof(float));
  FlexData sb = FlexData::make_object();
  sb.as_object().insert_or_assign("fps", FlexData::make_real(_fps));
  sb.as_object().insert_or_assign("frames",
                                  FlexData::make_int((std::int64_t)_frames));
  out->sideband = std::move(sb);
  ++_emitted;
  session()->info(fmt(
      "GenerateVideoStage('{}'): emitted latent video #{} [{}, {}, {}, {}] "
      "({} frames at {}x{}, {:.3f} fps)", this->id(), _emitted, ZC, T, h8, w8,
      _frames, _width, _height, _fps));
  co_await ctx.write(0, std::move(out));
}

#else   // !VPIPE_BUILD_APPLE_SILICON

Job
GenerateVideoStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  if (session()) {
    session()->error(fmt(
        "GenerateVideoStage('{}'): built without VPIPE_BUILD_APPLE_SILICON; "
        "the metal Wan DiT is unavailable, the stage is inert", this->id()));
  }
  co_return;
}

Job
GenerateVideoStage::process(RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  (void)in;
  ctx.signal_done();
  co_return;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

VPIPE_REGISTER_STAGE(GenerateVideoStage)
VPIPE_REGISTER_SPEC(GenerateVideoStage, kSpec)

}  // namespace vpipe
