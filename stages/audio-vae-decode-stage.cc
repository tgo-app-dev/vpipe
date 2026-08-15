#include "stages/audio-vae-decode-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "generative-models/shared/comfy-checkpoint.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-registry.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

// The model iport, appended after the primary `latent` input.
[[maybe_unused]] constexpr unsigned kModelPort = 1;

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "model dir carrying an audio VAE (read from <hf_dir>/audio_vae). "
          "OPTIONAL: a model-select source on the model iport overrides it",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "minimax-h3-fl2va"},
  {.key = "unload_when_idle", .type = ConfigType::String, .required = false,
   .doc = "drop the VAE weights after each beat and reload on the next one. "
          "\"auto\" (default) decides from physical RAM vs the pipeline's "
          "weight bytes; \"always\" / \"never\" force it",
   .def_str = "auto"},
};
const PortSpec kIports[] = {
  {.name = "latent",
   .doc = "f32 [stereo, latent_channels, frames] latent audio (whitened)",
   .type = &typeid(TensorBeatPayload),
   .tags = "latent", .clock_group = 0},
  {.name = "model",
   .doc = "OPTIONAL shared model reference from a model-select source; "
          "overrides the hf_dir config",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "audio",
   .doc = "decoded PCM as an f32 TensorBeat [channels, n_samples], planar",
   .type = &typeid(TensorBeatPayload),
   .tags = "audio-pcm", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "audio-vae-decode",
  .doc       = "Decodes a latent soundtrack into PCM on the metal-compute "
               "backend. The audio counterpart of vae-decode, and the "
               "downstream of generate-video's audio oport.",
  .display_name = "Audio VAE Decode",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

#ifdef VPIPE_BUILD_APPLE_SILICON
// Detect the family from the audio VAE's config.json `_class_name`.
// Empty when the directory holds no audio VAE at all, which is the
// common misconfiguration (an hf_dir pointing at a model that generates
// no soundtrack) and deserves a different message than a bad one.
std::string
audio_vae_family_(const std::string& vae_dir)
{
  // A Comfy-Org repack has no config.json anywhere: resolve_vae_dir hands
  // back the .safetensors itself and the architecture is named by that
  // file's `__metadata__` key. Checking it first is what keeps this stage
  // from reporting "holds no audio VAE" for a checkpoint that plainly has
  // one -- the resolver had already found it, and only the detection
  // below assumed a directory.
  if (genai::comfy::is_component(vae_dir, "minimax_h3_audio_vae")) {
    return "minimax-h3";
  }
  std::ifstream in(std::filesystem::path(vae_dir) / "config.json");
  if (!in) { return {}; }
  FlexData fd;
  try {
    fd = FlexData::from_json(in);
  } catch (...) {
    return {};
  }
  if (!fd.is_object()) { return {}; }
  auto obj = fd.as_object();
  if (!obj.contains("_class_name")) { return {}; }
  const std::string cls(obj.at("_class_name").as_string(""));
  if (cls == "MiniMaxH3AudioVAE") { return "minimax-h3"; }
  return {};
}
#endif

}  // namespace

AudioVaeDecodeStage::AudioVaeDecodeStage(const SessionContextIntf* s,
                                         std::string               id,
                                         std::vector<InEdge>       iports,
                                         FlexData                  config)
  : TypedStage<AudioVaeDecodeStage>(s, std::move(id), std::move(iports),
                                    std::move(config))
{
  // Deferred validation: construct for any config so a graph can be
  // edited before hf_dir exists. The "no model at all" case is reported
  // at initialize()/process(), when iport connectivity is known.
  _hf_dir = attr_str("hf_dir");
#ifdef VPIPE_BUILD_APPLE_SILICON
  {
    bool bad = false;
    _unload_cfg = model_memory::parse_unload_policy(
        attr_str("unload_when_idle"), &bad);
    if (bad) {
      session()->warn(fmt(
          "AudioVaeDecodeStage('{}'): unload_when_idle '{}' is not "
          "auto|always|never; using auto", this->id(),
          attr_str("unload_when_idle")));
    }
  }
#endif
  allocate_oports(spec().oports.size());
}

AudioVaeDecodeStage::~AudioVaeDecodeStage() = default;

const StageSpec&
AudioVaeDecodeStage::spec() const noexcept
{
  return kSpec;
}

#ifdef VPIPE_BUILD_APPLE_SILICON

void
AudioVaeDecodeStage::reset_run_state()
{
  if (!_h3_vae) {
    _load_attempted = false;
    _unloaded       = false;
  }
  _model_latched = false;
}

std::vector<ResourceClaim>
AudioVaeDecodeStage::declare_resources() const
{
  if (_hf_dir.empty()) { return {}; }
  const std::string root = resolve_model_dir(session(), _hf_dir);
  const std::string vae_dir =
      genai::MetalMiniMaxH3AudioVae::resolve_vae_dir(root);
  // The registry, not `_vae_family`: this is const and runs before
  // ensure_loaded_ has latched anything.
  if (genai::VaeModelFamily* f = genai::VaeModelRegistry::get().claim_for(
          session(), root, vae_dir,
          resolve_model(session(), _hf_dir).model_type)) {
    return f->declare_resources(root, vae_dir);
  }
  return model_memory::weight_claims({vae_dir});
}

void
AudioVaeDecodeStage::unload_vae_()
{
  if (!_h3_vae) { return; }
  _h3_vae.reset();
  _plugin_dec.reset();
  _unloaded = true;
  session()->log_debug(fmt(
      "AudioVaeDecodeStage('{}'): audio VAE unloaded (idle)", this->id()));
}

void
AudioVaeDecodeStage::reload_vae_()
{
  if (!_unloaded) { return; }
  _unloaded = false;
  _load_attempted = false;
  ensure_loaded_();
}

void
AudioVaeDecodeStage::ensure_loaded_()
{
  if (_load_attempted) { return; }
  _load_attempted = true;
  if (_hf_dir.empty()) {
    session()->error(fmt(
        "AudioVaeDecodeStage('{}'): no model -- set config.hf_dir or wire a "
        "model-select source to the model iport; inert", this->id()));
    return;
  }
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) {
    session()->error(fmt(
        "AudioVaeDecodeStage('{}'): no metal-compute backend on this "
        "session; the stage is inert", this->id()));
    return;
  }
  const std::string root = resolve_model_dir(session(), _hf_dir);
  const std::string vae_dir =
      genai::MetalMiniMaxH3AudioVae::resolve_vae_dir(root);
  // An out-of-tree family FIRST, and before audio_vae_family_ below:
  // that resolver looks for `audio_vae/config.json`, which a Comfy-style
  // pack does not have -- LTX-2.5 keeps its audio VAE as a bare
  // `vae/*audio-vae*.safetensors` with the config in `__metadata__`.
  if (!_family_probed) {
    _family_probed = true;
    _vae_family = genai::VaeModelRegistry::get().claim_for(
        session(), root, vae_dir,
        resolve_model(session(), _hf_dir).model_type);
  }
  if (_vae_family != nullptr) {
    _family = std::string(_vae_family->tag());
    genai::VaeModelCreateArgs args;
    args.root       = root;
    args.vae_dir    = vae_dir;
    args.model_type = resolve_model(session(), _hf_dir).model_type;
    args.metal      = mc;
    args.session    = session();
    try {
      _plugin_dec = _vae_family->load_audio_decoder(args);
    } catch (const std::exception& e) {
      session()->warn(fmt(
          "AudioVaeDecodeStage('{}'): VAE family '{}' threw loading '{}': "
          "{}; inert", this->id(), _family, root, e.what()));
      return;
    } catch (...) {
      session()->warn(fmt(
          "AudioVaeDecodeStage('{}'): VAE family '{}' threw a non-standard "
          "exception loading '{}'; inert", this->id(), _family, root));
      return;
    }
    if (!_plugin_dec) {
      // The family declined -- it generates no soundtrack, or has not
      // ported its audio VAE. Either way the stage stays inert rather
      // than emitting silence.
      session()->warn(fmt(
          "AudioVaeDecodeStage('{}'): family '{}' has no audio decoder; "
          "inert", this->id(), _family));
      return;
    }
    session()->info(fmt(
        "AudioVaeDecodeStage('{}'): audio VAE family '{}' loaded ({} Hz, "
        "{:.1f} MB)", this->id(), _family, _plugin_dec->sample_rate(),
        (double)_plugin_dec->resident_bytes() / (1024.0 * 1024.0)));
    return;                        // NOT the built-in chain below
  }

  _family = audio_vae_family_(vae_dir);
  if (_family.empty()) {
    session()->error(fmt(
        "AudioVaeDecodeStage('{}'): '{}' holds no audio VAE this stage "
        "knows (looked for audio_vae/config.json with a recognised "
        "_class_name); inert. A model that generates no soundtrack does "
        "not need this stage", this->id(), root));
    return;
  }

  // Idle-unload decision. Size the box against the PEERS that are
  // resident during a run -- the 33B transformer and the text encoder --
  // not against this VAE, which is ~60M parameters. It is idle for the
  // entire denoise and only wakes at the end.
  if (!_unload_resolved) {
    _unload_resolved = true;
    namespace fs = std::filesystem;
    const std::vector<std::string> peers = {
        (fs::path(root) / "transformer").string(),
        (fs::path(root) / "text_encoder").string()};
    switch (_unload_cfg) {
      case model_memory::UnloadPolicy::kAlways: _unload_idle = true;  break;
      case model_memory::UnloadPolicy::kNever:  _unload_idle = false; break;
      default:
        _unload_idle =
            model_memory::bounded(session(), peers, model_memory::kHeadroom);
        break;
    }
  }

  genai::MetalMiniMaxH3AudioVae::Config cfg;
  std::string err;
  if (!genai::MetalMiniMaxH3AudioVae::config_from_json(vae_dir, cfg, &err)) {
    session()->error(fmt(
        "AudioVaeDecodeStage('{}'): audio VAE config: {}; inert",
        this->id(), err));
    return;
  }
  // The whitening is what the DiT's latent space is expressed in. It
  // fails SILENTLY into identity if absent -- the decode still runs and
  // still produces a soundtrack, just one at the wrong per-channel
  // scale and offset -- so refuse rather than emit it.
  if ((int)cfg.latents_mean.size() != cfg.latent_channels ||
      (int)cfg.latents_std.size() != cfg.latent_channels) {
    session()->error(fmt(
        "AudioVaeDecodeStage('{}'): audio VAE config.json has {}/{} "
        "latents_mean/std for {} latent channels; without them a decode "
        "would silently run in the wrong latent space. Inert",
        this->id(), cfg.latents_mean.size(), cfg.latents_std.size(),
        cfg.latent_channels));
    return;
  }
  _h3_vae = genai::MetalMiniMaxH3AudioVae::load(vae_dir, mc, cfg);
  if (!_h3_vae) {
    // Naming the likely cause, because the config parsed a moment ago and
    // "could not load" alone points at the path -- which by here is known
    // good. Comfy-Org's repack of this component is not a re-encoding of
    // the released tensors but a DIFFERENT serialization: the weight-norm
    // parametrization is folded (no `.weight_v`) and the conv stack is
    // nested under `decoder.` / `encoder.` instead of the flat `conv_pre`
    // / `conv_post` this loader reads. That is a port, not a path, so it
    // fails here rather than being guessed at.
    const bool comfy =
        genai::comfy::is_component(vae_dir, "minimax_h3_audio_vae");
    session()->error(fmt(
        "AudioVaeDecodeStage('{}'): could not load the MiniMax-H3 audio "
        "VAE from '{}'; inert{}", this->id(), vae_dir,
        comfy ? " -- this is a Comfy-Org repack, whose audio VAE uses a "
                "different tensor naming (folded weight-norm, decoder.* / "
                "encoder.* prefixes) than the released checkpoint this "
                "loader reads. The video path is unaffected; generate "
                "without audio, or point hf_dir at the MiniMaxAI release"
              : ""));
    return;
  }
  session()->info(fmt(
      "AudioVaeDecodeStage('{}'): {} audio VAE loaded -- {} Hz, {} latent "
      "channels, hop {} ({} latents/s), unload_when_idle={}", this->id(),
      _family, cfg.sample_rate, cfg.latent_channels, cfg.hop(),
      cfg.hop() > 0 ? cfg.sample_rate / cfg.hop() : 0,
      _unload_idle ? "yes" : "no"));
}

Job
AudioVaeDecodeStage::initialize(RuntimeContext& ctx)
{
  const bool model_from_iport =
      ctx.num_iports() > kModelPort && ctx.iport_connected(kModelPort);
  if (!model_from_iport) { ensure_loaded_(); }
  co_return;
}

Job
AudioVaeDecodeStage::process(RuntimeContext& ctx)
{
  if (!_model_latched && ctx.num_iports() > kModelPort &&
      ctx.iport_connected(kModelPort)) {
    auto mb = co_await ctx.read(kModelPort);
    _model_latched = true;
    if (const auto* mfd =
            mb ? dynamic_cast<const FlexDataPayload*>(mb.get()) : nullptr) {
      if (apply_model_select_beat(mfd->data, _hf_dir)) { ensure_loaded_(); }
    }
  }
  auto in = co_await ctx.read(0);
  if (!in) { ctx.signal_done(); co_return; }
  if (_unloaded) { reload_vae_(); }

  // ---- an out-of-tree family --------------------------------------------
  //
  // Guarded on the POINTER and placed before the built-in branch, so a
  // family that tagged itself like a built-in still cannot fall through.
  // The FAMILY owns un-whitening, the vocoder and the sample rate; the
  // stage keeps the port, the beat and the sideband -- the same split
  // vae-decode draws.
  if (_vae_family != nullptr) {
    const auto* atb = dynamic_cast<const TensorBeatPayload*>(in.get());
    if (!_plugin_dec) {
      session()->warn(fmt(
          "AudioVaeDecodeStage('{}'): family '{}' has no audio decoder; "
          "skipping", this->id(), _family));
      co_return;
    }
    if (atb == nullptr || atb->dtype != TensorBeat::DType::F32 ||
        atb->shape.size() != 3) {
      session()->warn(fmt(
          "AudioVaeDecodeStage('{}'): expected an f32 3-D latent, got {}; "
          "skipping", this->id(), in->describe()));
      co_return;
    }
    genai::AudioVaeDecodeRequest areq;
    areq.latent = atb->as_f32();
    areq.shape.reserve(atb->shape.size());
    for (std::int64_t d : atb->shape) { areq.shape.push_back((int)d); }
    areq.sideband = &atb->sideband;
    if (atb->sideband.is_object()) {
      FlexData sb = atb->sideband;          // as_object() is a view
      auto o = sb.as_object();
      if (o.contains("latents_per_second")) {
        areq.latents_per_second = o.at("latents_per_second").as_real(0.0);
      }
    }
    areq.progress = [&ctx](int, int) { return !ctx.stop_requested(); };

    std::vector<float> apcm;
    std::vector<int> ashape;
    std::string aerr;
    bool aok = false;
    try {
      aok = _plugin_dec->decode(areq, &apcm, &ashape, &aerr);
    } catch (const std::exception& e) {
      session()->warn(fmt(
          "AudioVaeDecodeStage('{}'): family '{}' threw decoding: {}; "
          "skipping", this->id(), _family, e.what()));
      co_return;
    } catch (...) {
      session()->warn(fmt(
          "AudioVaeDecodeStage('{}'): family '{}' threw a non-standard "
          "exception decoding; skipping", this->id(), _family));
      co_return;
    }
    if (!aok || ashape.size() != 2) {
      session()->warn(fmt(
          "AudioVaeDecodeStage('{}'): decode failed ({}); skipping",
          this->id(), aerr.empty() ? "unknown error" : aerr));
      co_return;
    }
    const int ach = ashape[0], asamp = ashape[1];
    const int arate = _plugin_dec->sample_rate();
    auto aout = std::make_unique<TensorBeatPayload>();
    aout->dtype = TensorBeat::DType::F32;
    aout->shape = {(std::int64_t)ach, (std::int64_t)asamp};
    aout->resize_contiguous(apcm.size());
    std::memcpy(aout->as_f32(), apcm.data(), apcm.size() * sizeof(float));
    {
      FlexData sb = FlexData::make_object();
      sb.as_object().insert_or_assign(
          "sample_rate", FlexData::make_int((std::int64_t)arate));
      sb.as_object().insert_or_assign(
          "channels", FlexData::make_int((std::int64_t)ach));
      sb.as_object().insert_or_assign(
          "samples", FlexData::make_int((std::int64_t)asamp));
      aout->sideband = std::move(sb);
    }
    ++_clips;
    session()->info(fmt(
        "AudioVaeDecodeStage('{}'): family '{}' decoded clip #{} -- {} x {} "
        "samples ({:.2f} s at {} Hz)", this->id(), _family, _clips, ach,
        asamp, arate > 0 ? (double)asamp / arate : 0.0, arate));
    if (_unload_idle) { unload_vae_(); }
    co_await ctx.write(0, std::move(aout));
    co_return;
  }

  if (!_h3_vae) {
    session()->warn(fmt(
        "AudioVaeDecodeStage('{}'): no audio VAE loaded; skipping",
        this->id()));
    co_return;
  }
  const auto* tbp = dynamic_cast<const TensorBeatPayload*>(in.get());
  if (tbp == nullptr || tbp->dtype != TensorBeat::DType::F32 ||
      tbp->shape.size() != 3) {
    session()->warn(fmt(
        "AudioVaeDecodeStage('{}'): expected an f32 [stereo, channels, "
        "frames] latent TensorBeat, got {}; skipping", this->id(),
        in->describe()));
    co_return;
  }
  const auto& c = _h3_vae->config();
  const int B  = (int)tbp->shape[0];
  const int ZC = (int)tbp->shape[1];
  const int T  = (int)tbp->shape[2];
  if (B <= 0 || T <= 0 || ZC != c.latent_channels) {
    session()->warn(fmt(
        "AudioVaeDecodeStage('{}'): latent [{}, {}, {}] does not match "
        "latent_channels {}; skipping", this->id(), B, ZC, T,
        c.latent_channels));
    co_return;
  }
  const std::size_t n = (std::size_t)B * ZC * T;
  if (tbp->element_count() < n) {
    session()->warn(fmt(
        "AudioVaeDecodeStage('{}'): latent beat holds {} elements, its "
        "shape needs {}; skipping", this->id(), tbp->element_count(), n));
    co_return;
  }

  // Un-whiten on the way in: the DiT generates in NORMALIZED latent
  // space, so the VAE has to be handed the latents it was trained on.
  std::vector<float> z(n);
  {
    const float* s = tbp->as_f32();
    for (int b = 0; b < B; ++b) {
      for (int ch = 0; ch < ZC; ++ch) {
        const float mu = c.latents_mean[(std::size_t)ch];
        const float sd = c.latents_std[(std::size_t)ch];
        const std::size_t off = ((std::size_t)b * ZC + ch) * T;
        for (int t = 0; t < T; ++t) {
          z[off + (std::size_t)t] = s[off + (std::size_t)t] * sd + mu;
        }
      }
    }
  }

  std::vector<float> pcm;
  std::string err;
  {
    PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmVae,
                       kPerfLlmVaeBegin,
                       (std::uint64_t)B * T * c.hop());
    if (!_h3_vae->decode(z.data(), B, T, &pcm, &err)) {
      session()->warn(fmt("AudioVaeDecodeStage('{}'): decode failed: {}",
                          this->id(), err));
      if (_unload_idle) { unload_vae_(); }
      co_return;
    }
  }
  const int samples = B > 0 ? (int)(pcm.size() / (std::size_t)B) : 0;

  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::F32;
  out->shape = {(std::int64_t)B, (std::int64_t)samples};
  out->resize_contiguous(pcm.size());
  std::memcpy(out->as_f32(), pcm.data(), pcm.size() * sizeof(float));
  {
    // The rate and the channel count both ride on the beat: a sink has
    // no other way to learn them (there is no ffmpeg header upstream of
    // a generated soundtrack), and a sink that guessed would resample or
    // downmix the whole clip silently.
    FlexData sb = FlexData::make_object();
    sb.as_object().insert_or_assign(
        "sample_rate", FlexData::make_int((std::int64_t)c.sample_rate));
    sb.as_object().insert_or_assign(
        "channels", FlexData::make_int((std::int64_t)B));
    sb.as_object().insert_or_assign(
        "samples", FlexData::make_int((std::int64_t)samples));
    out->sideband = std::move(sb);
  }
  ++_clips;
  session()->info(fmt(
      "AudioVaeDecodeStage('{}'): decoded clip #{} -- {} latent frames -> "
      "{} x {} samples ({:.2f} s at {} Hz)", this->id(), _clips, T, B,
      samples, c.sample_rate > 0 ? (double)samples / c.sample_rate : 0.0,
      c.sample_rate));
  if (_unload_idle) { unload_vae_(); }
  co_await ctx.write(0, std::move(out));
  co_return;
}

#else   // !VPIPE_BUILD_APPLE_SILICON

void AudioVaeDecodeStage::reset_run_state() {}

std::vector<ResourceClaim>
AudioVaeDecodeStage::declare_resources() const
{
  return {};
}

Job
AudioVaeDecodeStage::initialize(RuntimeContext&)
{
  session()->error(fmt(
      "AudioVaeDecodeStage('{}'): built without the apple-silicon backend; "
      "the stage is inert", this->id()));
  co_return;
}

Job
AudioVaeDecodeStage::process(RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  (void)in;
  ctx.signal_done();
  co_return;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

VPIPE_REGISTER_STAGE(AudioVaeDecodeStage)
VPIPE_REGISTER_SPEC(AudioVaeDecodeStage, kSpec)

}  // namespace vpipe
