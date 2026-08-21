#include "stages/audio-vae-encode-stage.h"
#include "generative-models/minimax-h3/metal-minimax-h3-audio-vae.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-registry.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

// The model iport, appended after the primary `audio` input.
[[maybe_unused]] constexpr unsigned kModelPort = 1;

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "model dir carrying an audio VAE. OPTIONAL: a model-select source "
          "on the model iport overrides it",
   .suggest_db = kModelRegistryDb},
  {.key = "sample_rate", .type = ConfigType::Int, .required = false,
   .doc = "FALLBACK sample rate, used only for a beat whose sideband names "
          "none. A producer that stamps its rate (audio-to-pcm does) makes "
          "this unnecessary"},
  {.key = "max_seconds", .type = ConfigType::Real, .required = false,
   .doc = "refuse to buffer more than this many seconds. A reference "
          "soundtrack is a clip, not a live feed, and an unbounded buffer "
          "on an open microphone fails as an OOM rather than a message",
   .def_str = "30"},
  {.key = "unload_when_idle", .type = ConfigType::String, .required = false,
   .doc = "drop the VAE weights after the encode. \"auto\" (default) decides "
          "from physical RAM vs the pipeline's weight bytes; "
          "\"always\" / \"never\" force it",
   .def_str = "auto"},
};
const PortSpec kIports[] = {
  {.name = "audio",
   .doc = "f32 PCM, rank-1 [N] mono or rank-2 [channels, N] PLANAR, in "
          "[-1, 1]; sideband.sample_rate names the rate. EVERY beat is "
          "accumulated and the encode runs ONCE at EOS",
   .type = &typeid(TensorBeatPayload),
   // BOTH spellings, because the tree has both and either can drive
   // this: `audio-to-pcm` tags its output "pcm-samples" and
   // `audio-vae-decode` tags its own "audio-pcm". Tags are OR-matched,
   // and a mismatch is a REFUSED EDGE at launch, not a warning.
   .tags = "pcm-samples, audio-pcm", .clock_group = 0},
  {.name = "model",
   .doc = "OPTIONAL shared model reference from a model-select source; "
          "overrides the hf_dir config",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "latent",
   .doc = "the encoded soundtrack as an f32 TensorBeat in the ENCODER's own "
          "latent shape -- [rows, dim] for a family feeding generate-video's "
          "ref_audio_rows. ONE beat per run, at drain",
   .type = &typeid(TensorBeatPayload),
   .tags = "latent", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "audio-vae-encode",
  .doc       = "Encodes PCM into a latent soundtrack -- the reference an "
               "omni generator conditions on. The audio counterpart of "
               "vae-encode, and the upstream of generate-video's "
               "ref_audio_rows. Accumulates every beat and encodes once at "
               "EOS: an audio VAE is causal and compresses time, so chunks "
               "encoded separately do not concatenate.",
  .display_name = "Audio VAE Encode",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

AudioVaeEncodeStage::AudioVaeEncodeStage(const SessionContextIntf* s,
                                         std::string               id,
                                         std::vector<InEdge>       iports,
                                         FlexData                  config)
  : TypedStage<AudioVaeEncodeStage>(s, std::move(id), std::move(iports),
                                    std::move(config))
{
  _hf_dir   = attr_str("hf_dir");
  _cfg_rate = (int)attr_int("sample_rate");
  {
    const double m = attr_real("max_seconds");
    if (m > 0.0) { _max_seconds = m; }
  }
#ifdef VPIPE_BUILD_APPLE_SILICON
  {
    bool bad = false;
    _unload_cfg = model_memory::parse_unload_policy(
        attr_str("unload_when_idle"), &bad);
    if (bad) {
      session()->warn(fmt(
          "AudioVaeEncodeStage('{}'): unload_when_idle '{}' is not "
          "auto|always|never; using auto", this->id(),
          attr_str("unload_when_idle")));
    }
  }
#endif
  allocate_oports(spec().oports.size());
}

AudioVaeEncodeStage::~AudioVaeEncodeStage() = default;

const StageSpec&
AudioVaeEncodeStage::spec() const noexcept
{
  return kSpec;
}

bool
AudioVaeEncodeStage::accumulate_(const TensorBeatPayload& tb)
{
  if (tb.dtype != TensorBeat::DType::F32) {
    session()->warn(fmt(
        "AudioVaeEncodeStage('{}'): PCM must be f32; dropping a beat",
        this->id()));
    return false;
  }
  int ch = 1, n = 0;
  if (tb.shape.size() == 1) {
    n = (int)tb.shape[0];
  } else if (tb.shape.size() == 2) {
    ch = (int)tb.shape[0];
    n  = (int)tb.shape[1];
  } else {
    session()->warn(fmt(
        "AudioVaeEncodeStage('{}'): expected rank-1 [N] or rank-2 "
        "[channels, N] PCM; dropping a beat of rank {}", this->id(),
        tb.shape.size()));
    return false;
  }
  if (ch <= 0 || n <= 0) { return false; }

  // The rate comes from the beat. A producer that stamps none falls back
  // to the config; a stream that CHANGES rate mid-way is refused rather
  // than spliced -- the result would be the right length and the wrong
  // pitch across the join, which no downstream check would catch.
  int rate = _cfg_rate;
  if (tb.sideband.is_object()) {
    FlexData sb = tb.sideband;              // as_object() returns a VIEW
    auto o = sb.as_object();
    if (o.contains("sample_rate")) {
      rate = (int)o.at("sample_rate").as_int(rate);
    }
  }
  if (_rate == 0) {
    _rate = rate;
    _pcm.assign((std::size_t)ch, {});
  } else if (rate != 0 && rate != _rate) {
    session()->warn(fmt(
        "AudioVaeEncodeStage('{}'): the stream changed from {} Hz to {} Hz "
        "mid-clip; dropping the beat rather than splicing two rates",
        this->id(), _rate, rate));
    return false;
  }
  if ((int)_pcm.size() != ch) {
    session()->warn(fmt(
        "AudioVaeEncodeStage('{}'): the stream changed from {} channels to "
        "{}; dropping the beat", this->id(), (int)_pcm.size(), ch));
    return false;
  }

  if (_rate > 0 && !_pcm.empty()) {
    const double have = (double)_pcm[0].size() / _rate;
    if (have + (double)n / _rate > _max_seconds) {
      if (!_overflowed) {
        _overflowed = true;
        session()->warn(fmt(
            "AudioVaeEncodeStage('{}'): the reference reached max_seconds "
            "({:.1f} s); every later sample is dropped. Raise max_seconds, "
            "or bound the source", this->id(), _max_seconds));
      }
      return false;
    }
  }

  const auto host = tb.materialize_contiguous();
  const float* p = reinterpret_cast<const float*>(host.data());
  for (int c = 0; c < ch; ++c) {
    _pcm[(std::size_t)c].insert(_pcm[(std::size_t)c].end(),
                                p + (std::size_t)c * n,
                                p + (std::size_t)(c + 1) * n);
  }
  return true;
}

void
AudioVaeEncodeStage::apply_constant(unsigned iport, const FlexData& beat)
{
  // Pre-launch twin of the runtime latch in process(): the same
  // beat and the same parse, early enough that declare_resources()
  // sees the model. Bookkeeping only -- nothing loads here; the
  // pipeline is not assembled yet (see Stage::apply_constant).
  if (iport != kModelPort) { return; }
  apply_model_select_beat(beat, _hf_dir);
}

#ifdef VPIPE_BUILD_APPLE_SILICON

void
AudioVaeEncodeStage::reset_run_state()
{
  _model_latched = false;
  _pcm.clear();
  _rate = 0;
  _overflowed = false;
}

StageMemory
AudioVaeEncodeStage::declare_memory() const
{
  StageMemory m;
  if (_hf_dir.empty()) { return m; }
  const std::string root = resolve_model_dir(session(), _hf_dir);
  // Through the audio VAE's OWN resolver, not the root. declare_resources
  // hands the family (root, root) and lets it answer; that works because
  // a family sizes itself, but a byte figure derived from `root` would
  // be the whole repository. When the component cannot be located,
  // nothing is the honest answer.
  std::string vae = genai::MetalMiniMaxH3AudioVae::resolve_vae_dir(root);
  // A registered family knows where its own audio VAE sits, which on a
  // single-file pack is the only way this is answerable at all.
  if (genai::VaeModelFamily* f = genai::VaeModelRegistry::get().claim_for(
          session(), root, vae,
          resolve_model(session(), _hf_dir).model_type)) {
    const std::string a =
        f->vae_path(root, genai::VaeModelFamily::Role::kAudio);
    if (!a.empty()) { vae = a; }
  }
  if (vae.empty() || vae == root) { return m; }
  // No streaming form; a VAE holds what it weighs.
  //
  // NEITHER `releases` NOR `reclaimable`, whatever the config says, and
  // the config is what makes that worth stating: this stage parses
  // `unload_when_idle` but has no unload path -- it never drops the
  // encoder. A declaration is a promise peers size against, so claiming
  // a release this stage does not perform would leave them short by the
  // encoder's whole size. Give it an unload path and these come back.
  m.hold(vae, model_memory::dir_weights_bytes(vae));
  return m;
}

std::vector<ResourceClaim>
AudioVaeEncodeStage::declare_resources() const
{
  if (_hf_dir.empty()) { return {}; }
  const std::string root = resolve_model_dir(session(), _hf_dir);
  if (genai::VaeModelFamily* f = genai::VaeModelRegistry::get().claim_for(
          session(), root, root,
          resolve_model(session(), _hf_dir).model_type)) {
    return f->declare_resources(root, root);
  }
  return {};
}

void
AudioVaeEncodeStage::ensure_loaded_()
{
  if (_load_attempted) { return; }
  _load_attempted = true;
  if (_hf_dir.empty()) {
    session()->error(fmt(
        "AudioVaeEncodeStage('{}'): no model -- set config.hf_dir or wire a "
        "model-select source to the model iport; inert", this->id()));
    return;
  }
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) {
    session()->error(fmt(
        "AudioVaeEncodeStage('{}'): no metal-compute backend on this "
        "session; the stage is inert", this->id()));
    return;
  }
  const std::string root = resolve_model_dir(session(), _hf_dir);
  const std::string model_type = resolve_model(session(), _hf_dir).model_type;
  if (!_family_probed) {
    _family_probed = true;
    // `root` for BOTH arguments: there is no built-in audio encoder to
    // share a resolver with, and every family this stage can reach keeps
    // its own layout. A family that wants the diffusers spelling
    // resolves it from `root` itself.
    _vae_family = genai::VaeModelRegistry::get().claim_for(
        session(), root, root, model_type);
  }
  if (_vae_family == nullptr) {
    session()->error(fmt(
        "AudioVaeEncodeStage('{}'): no registered VAE family claims '{}'. "
        "This stage has NO built-in encoder -- every audio encoder comes "
        "from a plugin -- so there is nothing to fall back to; inert",
        this->id(), root));
    return;
  }
  _family = std::string(_vae_family->tag());
  genai::VaeModelCreateArgs args;
  args.root       = root;
  args.vae_dir    = root;
  args.model_type = model_type;
  args.metal      = mc;
  args.session    = session();
  try {
    _plugin_enc = _vae_family->load_audio_encoder(args);
  } catch (const std::exception& e) {
    session()->warn(fmt(
        "AudioVaeEncodeStage('{}'): VAE family '{}' threw loading '{}': {}; "
        "inert", this->id(), _family, root, e.what()));
    return;
  } catch (...) {
    session()->warn(fmt(
        "AudioVaeEncodeStage('{}'): VAE family '{}' threw a non-standard "
        "exception loading '{}'; inert", this->id(), _family, root));
    return;
  }
  if (!_plugin_enc) {
    session()->error(fmt(
        "AudioVaeEncodeStage('{}'): family '{}' has no audio ENCODER (it may "
        "decode only); inert", this->id(), _family));
    return;
  }
  if (!_unload_idle) {
    switch (_unload_cfg) {
      case model_memory::UnloadPolicy::kAlways: _unload_idle = true;  break;
      case model_memory::UnloadPolicy::kNever:  _unload_idle = false; break;
      default:
        _unload_idle = model_memory::bounded(session(), {root},
                                             model_memory::kHeadroom);
        break;
    }
  }
  session()->info(fmt(
      "AudioVaeEncodeStage('{}'): audio VAE family '{}' loaded -- wants {} Hz "
      "x {} channel(s), {:.1f} MB", this->id(), _family,
      _plugin_enc->sample_rate(), _plugin_enc->channels(),
      (double)_plugin_enc->resident_bytes() / (1024.0 * 1024.0)));
}

Job
AudioVaeEncodeStage::initialize(RuntimeContext& ctx)
{
  const bool model_from_iport =
      ctx.num_iports() > kModelPort && ctx.iport_connected(kModelPort);
  if (!model_from_iport) { ensure_loaded_(); }
  co_return;
}

Job
AudioVaeEncodeStage::process(RuntimeContext& ctx)
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
  if (in) {
    // Still streaming: buffer and wait. The encode needs the WHOLE clip
    // (see the header), so nothing is emitted per beat.
    if (const auto* tb = dynamic_cast<const TensorBeatPayload*>(in.get())) {
      accumulate_(*tb);
    } else {
      session()->warn(fmt(
          "AudioVaeEncodeStage('{}'): expected a PCM TensorBeat, got {}; "
          "skipping", this->id(), in->describe()));
    }
    co_return;
  }

  // ---- EOS: encode what was gathered, then close -------------------------
  const int ch = (int)_pcm.size();
  const int n  = ch > 0 ? (int)_pcm[0].size() : 0;
  if (ch == 0 || n == 0) {
    session()->warn(fmt(
        "AudioVaeEncodeStage('{}'): the stream ended with no PCM; no "
        "reference emitted", this->id()));
    ctx.signal_done();
    co_return;
  }
  ensure_loaded_();
  if (!_plugin_enc) {
    ctx.signal_done();
    co_return;
  }
  const int want_rate = _plugin_enc->sample_rate();
  if (_rate <= 0) {
    session()->warn(fmt(
        "AudioVaeEncodeStage('{}'): no beat named a sample rate and none is "
        "configured; refusing to guess. Set config.sample_rate, or use a "
        "producer that stamps it", this->id()));
    ctx.signal_done();
    co_return;
  }
  if (want_rate > 0 && _rate != want_rate) {
    // REFUSED, not resampled. This stage owns no resampler, and encoding
    // 48 kHz samples as though they were 16 kHz produces a reference of
    // the right length at the wrong pitch -- which nothing downstream can
    // see.
    session()->error(fmt(
        "AudioVaeEncodeStage('{}'): the '{}' encoder wants {} Hz and the "
        "stream is {} Hz. Set audio-to-pcm's output_sample_rate to {}; this "
        "stage does not resample, because a silent rate mismatch is a "
        "reference of the right length at the wrong pitch", this->id(),
        _family, want_rate, _rate, want_rate));
    ctx.signal_done();
    co_return;
  }

  // Flatten to the planar block the family takes.
  std::vector<float> flat((std::size_t)ch * n);
  for (int c = 0; c < ch; ++c) {
    std::memcpy(flat.data() + (std::size_t)c * n, _pcm[(std::size_t)c].data(),
                (std::size_t)n * sizeof(float));
  }
  genai::AudioVaeEncodeRequest req;
  req.pcm         = flat.data();
  req.channels    = ch;
  req.n_samples   = n;
  req.sample_rate = _rate;

  std::vector<float> lat;
  std::vector<int> shape;
  std::string err;
  bool ok = false;
  {
    PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmVae, kPerfLlmVaeBegin,
                       (std::uint64_t)n);
    try {
      ok = _plugin_enc->encode(req, &lat, &shape, &err);
    } catch (const std::exception& e) {
      session()->warn(fmt(
          "AudioVaeEncodeStage('{}'): family '{}' threw encoding: {}",
          this->id(), _family, e.what()));
      ctx.signal_done();
      co_return;
    } catch (...) {
      session()->warn(fmt(
          "AudioVaeEncodeStage('{}'): family '{}' threw a non-standard "
          "exception encoding", this->id(), _family));
      ctx.signal_done();
      co_return;
    }
  }
  if (!ok || lat.empty() || shape.empty()) {
    session()->warn(fmt(
        "AudioVaeEncodeStage('{}'): the {} encoder produced nothing ({})",
        this->id(), _family, err.empty() ? "no reason given" : err));
    ctx.signal_done();
    co_return;
  }
  std::size_t want = 1;
  for (int d : shape) { want *= (std::size_t)std::max(0, d); }
  if (want != lat.size()) {
    session()->warn(fmt(
        "AudioVaeEncodeStage('{}'): the {} encoder returned {} values for a "
        "shape holding {}", this->id(), _family, lat.size(), want));
    ctx.signal_done();
    co_return;
  }

  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::F32;
  out->shape.assign(shape.begin(), shape.end());
  out->resize_contiguous(lat.size());
  std::memcpy(out->as_f32(), lat.data(), lat.size() * sizeof(float));
  {
    FlexData sb = FlexData::make_object();
    sb.as_object().insert_or_assign(
        "sample_rate", FlexData::make_int((std::int64_t)_rate));
    sb.as_object().insert_or_assign(
        "seconds", FlexData::make_real((double)n / _rate));
    out->sideband = std::move(sb);
  }
  ++_latents;
  std::string dims;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    dims += (i ? ", " : "") + std::to_string(shape[i]);
  }
  session()->info(fmt(
      "AudioVaeEncodeStage('{}'): {} encoded a {:.2f} s reference ({} x {} "
      "samples at {} Hz) -> latent [{}]", this->id(), _family,
      (double)n / _rate, ch, n, _rate, dims));
  if (_unload_idle) { _plugin_enc->release_idle(); }
  co_await ctx.write(0, std::move(out));
  // The reference is ONE beat and the stream is over; closing here is
  // what lets a downstream generate-video see EOS on the port rather
  // than waiting for a second reference that never comes.
  ctx.signal_done();
}

#else   // !VPIPE_BUILD_APPLE_SILICON

void AudioVaeEncodeStage::reset_run_state() { _pcm.clear(); _rate = 0; }

std::vector<ResourceClaim>
AudioVaeEncodeStage::declare_resources() const { return {}; }

Job
AudioVaeEncodeStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  if (session()) {
    session()->error(fmt(
        "AudioVaeEncodeStage('{}'): built without VPIPE_BUILD_APPLE_SILICON; "
        "the metal VAE is unavailable, the stage is inert", this->id()));
  }
  co_return;
}

Job
AudioVaeEncodeStage::process(RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  (void)in;
  ctx.signal_done();
  co_return;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

VPIPE_REGISTER_STAGE(AudioVaeEncodeStage)
VPIPE_REGISTER_SPEC(AudioVaeEncodeStage, kSpec)

}  // namespace vpipe
