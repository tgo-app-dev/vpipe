#include "stages/video-ref-encoder-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/ffmpeg-libraries.h"
#include "common/media-decode.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#endif

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

// The model iport (a model-select source) overrides hf_dir. It follows
// the prompt. (Referenced only from the Apple-gated code.)
[[maybe_unused]] constexpr unsigned kModelPort = 1;

const ConfigKey kAttrs[] = {
  {.key = "references", .type = ConfigType::Any, .required = true,
   .doc = "the reference files IN THE ORDER THE MODEL SHOULD READ THEM: one "
          "path, or an array of paths. The order labels them in the "
          "presentation (<Picture 1>, <Video 2>, ...) and lays them out on "
          "the shared rotary clock, so a different order is a different "
          "request. WHAT each file is -- image, video or audio -- is read "
          "from the FILE, not from its name: a container with one frame is "
          "an image reference, an .mp4 with no video stream is an audio one. "
          "At most 9 images, 3 videos and 3 audios, 12 in total, and audio "
          "cannot be the only kind",
   .is_path = true, .path_filter = "image,video,audio"},
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "the MiniMax-H3 model dir (text_encoder/, video_vae/, audio_vae/). "
          "OPTIONAL: a model-select source on the model iport overrides it",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "minimax-h3-ref2va"},
  {.key = "frames", .type = ConfigType::Int, .required = false,
   .doc = "the GENERATED frame count at 24 fps, snapped up to the next "
          "17n+5. MUST MATCH the generate-video stage's `frames`: it is the "
          "duration every reference is truncated to, and the layout the DiT "
          "builds is sized from the same number",
   .def_int = 121},
  {.key = "reference_image_short_edge", .type = ConfigType::Int,
   .required = false,
   .doc = "what an image reference's short edge is scaled to (2048 for the "
          "released checkpoint). Unlike the target canvas this has NO area "
          "cap and upscales a small picture -- an image reference is read at "
          "high detail and never binds the generated geometry",
   .def_int = 2048},
  {.key = "video_sample_fps", .type = ConfigType::Real, .required = false,
   .doc = "the rate the CONDITIONER reads a video reference at -- every "
          "24/this-th of the normalized frames, merged in pairs into "
          "timestamped vision blocks. Not the rate the VAE encodes it at, "
          "which is the full 24",
   .def_real = 2.0},
  {.key = "max_prompt_tokens", .type = ConfigType::Int, .required = false,
   .doc = "the conditioner's sequence pool. A ref2va presentation is FAR "
          "longer than a text-only prompt -- a single 2048-short-edge image "
          "reference contributes thousands of vision tokens -- so the "
          "text-path default of 4096 does not fit one. Costs KV: the tapped "
          "50 layers are ~200 KB a token, so 16384 is ~3.3 GB held while the "
          "conditioner is resident. Raise it for a request with many image "
          "references; lower it on a small box",
   .def_int = 16384},
  {.key = "unload_when_idle", .type = ConfigType::String, .required = false,
   .doc = "drop the conditioner and the VAEs once the references are encoded "
          "and reload them for the next request. The 32B conditioner is the "
          "largest resident block in a ref2va graph and is idle for the whole "
          "denoise, which on this model is minutes. \"auto\" (default) decides "
          "from physical RAM vs the pipeline's weight bytes; \"always\" / "
          "\"never\" force it",
   .def_str = "auto"},
};

const PortSpec kIports[] = {
  {.name = "prompt",
   .doc = "prompt text (FlexData string or {text: ...}), used VERBATIM -- no "
          "chat template and no special tokens",
   .type = &typeid(FlexDataPayload), .tags = "text", .clock_group = 0},
  {.name = "model",
   .doc = "OPTIONAL shared model reference from a model-select source; "
          "overrides the hf_dir config",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};

const PortSpec kOports[] = {
  {.name = "conditioning",
   .doc = "bf16 [n_tokens, 5120] -- the Qwen3-VL-32B layer-50 tap over the "
          "whole presentation. The same contract diffusion-conditioner "
          "emits, so generate-video takes either; the SIDEBAND carries the "
          "per-row modality tags and every reference's latent geometry",
   .type = &typeid(TensorBeatPayload), .tags = "conditioning",
   .clock_group = 0},
  {.name = "ref_video",
   .doc = "f32 [rows, 96] -- the image and video references' latents packed "
          "into DiT rows, concatenated in reference order. 0 rows when no "
          "reference carries video",
   .type = &typeid(TensorBeatPayload), .tags = "latent", .clock_group = 0},
  {.name = "ref_audio",
   .doc = "f32 [rows, 32] -- the reference soundtracks, channel-major within "
          "a reference and in the same order. 0 rows when none",
   .type = &typeid(TensorBeatPayload), .tags = "latent", .clock_group = 0},
};

const StageSpec kSpec = {
  .type_name = "video-ref-encoder",
  .doc       = "MiniMax-H3 ref2va reference encoder: a list of reference "
               "images, clips and soundtracks (+ a prompt) -> the "
               "conditioning and the reference latent rows the video DiT "
               "denoises against. The list is the input rather than a port "
               "per reference because a request carries a VARIABLE dozen of "
               "them, and each is decoded here so its frame rate and sample "
               "rate reach the model that resamples onto 24 fps. Pair it "
               "with a generate-video stage on the same hf_dir and the same "
               "`frames`.",
  .display_name = "Video Reference Encoder",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

#ifdef VPIPE_BUILD_APPLE_SILICON

namespace h3 = genai::minimax_h3;

// Look a key up in a FlexData object, BY VALUE.
//
// as_object() hands back a view whose iterators dereference to a freshly
// built pair, so `it->second` is not addressable and a span or a
// string_view taken through one dangles. at() returns a value, which is
// what the caller then has to bind to a local.
bool
obj_get_(const FlexData& d, const char* key, FlexData* out)
{
  if (!d.is_object()) { return false; }
  const auto o = d.as_object();
  if (!o.contains(key)) { return false; }
  *out = o.at(key);
  return true;
}

// Read the prompt out of a FlexData beat: a bare string, or an object
// with `text` / `prompt`.
std::string
prompt_of_(const FlexData& d)
{
  if (d.is_string()) { return std::string(d.as_string()); }
  for (const char* k : {"text", "prompt"}) {
    FlexData v;
    if (obj_get_(d, k, &v) && v.is_string()) {
      return std::string(v.as_string());
    }
  }
  return {};
}

#endif  // VPIPE_BUILD_APPLE_SILICON

}  // namespace

VideoRefEncoderStage::VideoRefEncoderStage(const SessionContextIntf* s,
                                           std::string               id,
                                           std::vector<InEdge>       iports,
                                           FlexData                  config)
  : TypedStage<VideoRefEncoderStage>(s, std::move(id), std::move(iports),
                                     std::move(config))
{
  // hf_dir is OPTIONAL: a model-select source can supply it instead, so
  // "no model at all" is reported at initialize()/process() time, when
  // iport connectivity is known, rather than at construction.
  _hf_dir           = attr_str("hf_dir");
  // One path or a list of them. The composer's file browser appends
  // into a JSON array, and a single pick left as a bare string is what
  // a hand-written pipeline looks like -- both are the same request, so
  // both are accepted rather than one being the spelling that works.
  {
    auto o = this->config().as_object();
    FlexData refs = o.contains("references") ? o.at("references")
                                             : FlexData::make_null();
    if (refs.is_string()) {
      _references.push_back(std::string(refs.as_string("")));
    } else if (refs.is_array()) {
      auto arr = refs.as_array();
      for (std::size_t i = 0; i < arr.size(); ++i) {
        if (arr[i].is_string()) {
          _references.push_back(std::string(arr[i].as_string("")));
        } else {
          fail_config(fmt(
              "VideoRefEncoderStage('{}'): references[{}] is not a path "
              "string", this->id(), i));
        }
      }
    } else if (!refs.is_null()) {
      fail_config(fmt(
          "VideoRefEncoderStage('{}'): references must be a path or an array "
          "of paths", this->id()));
    }
    // Empty is a config error and not a runtime warning: a `ref2va`
    // checkpoint with no references generates video conditioned on
    // nothing, at full cost. Better to refuse the graph at launch.
    if (_references.empty() && config_error().empty()) {
      fail_config(fmt(
          "VideoRefEncoderStage('{}'): references is required and must name "
          "at least one file", this->id()));
    }
  }
  _frames           = attr_int("frames");
  _ref_short_edge   = attr_int("reference_image_short_edge");
  _video_sample_fps = attr_real("video_sample_fps");
  _max_prompt_tokens = attr_int("max_prompt_tokens");
#ifdef VPIPE_BUILD_APPLE_SILICON
  {
    bool bad = false;
    _unload_cfg = model_memory::parse_unload_policy(
        attr_str("unload_when_idle"), &bad);
    if (bad) {
      session()->warn(fmt(
          "VideoRefEncoderStage('{}'): unload_when_idle '{}' is not "
          "auto|always|never; using auto", this->id(),
          attr_str("unload_when_idle")));
    }
  }
#endif
  allocate_oports(spec().oports.size());
}

VideoRefEncoderStage::~VideoRefEncoderStage() = default;

const StageSpec&
VideoRefEncoderStage::spec() const noexcept
{
  return kSpec;
}

void
VideoRefEncoderStage::apply_constant(unsigned iport, const FlexData& beat)
{
  // Pre-launch twin of the runtime latch in process(): the same
  // beat and the same parse, early enough that declare_resources()
  // sees the model. Bookkeeping only -- nothing loads here; the
  // pipeline is not assembled yet (see Stage::apply_constant).
  if (iport != kModelPort) { return; }
  apply_model_select_beat(beat, _hf_dir);
}

#ifndef VPIPE_BUILD_APPLE_SILICON

Job VideoRefEncoderStage::initialize(RuntimeContext&) { co_return; }
Job VideoRefEncoderStage::process(RuntimeContext&) { co_return; }
void VideoRefEncoderStage::reset_run_state() {}
std::vector<ResourceClaim>
VideoRefEncoderStage::declare_resources() const { return {}; }

#else

void
VideoRefEncoderStage::reset_run_state()
{
  // Per-launch reset: the stage survives a stop/relaunch and the select
  // sources upstream re-emit on every launch. Without this the re-emitted
  // beat is never latched and the stage keeps the previous run's model.
  _model_latched   = false;
  _unload_resolved = false;
}

StageMemory
VideoRefEncoderStage::declare_memory() const
{
  StageMemory m;
  if (_hf_dir.empty()) { return m; }
  namespace fs = std::filesystem;
  const std::string root = resolve_model_dir(session(), _hf_dir);
  std::string enc = genai::MiniMaxH3TextEncoder::resolve_encoder_dir(root);
  if (enc.empty()) { enc = (fs::path(root) / "text_encoder").string(); }
  std::string vvae = genai::MetalMiniMaxH3VideoVae::resolve_vae_dir(root);
  if (vvae.empty()) { vvae = (fs::path(root) / "video_vae").string(); }
  // ONLY what this stage HOLDS.
  //
  // declare_resources also names the DiT, because a claim is what every
  // peer sizes itself against and a 66 GB component nobody declared is
  // the silent under-count the planning phase exists to prevent. This
  // is a different question: StageMemory describes one stage's own
  // holdings, and the DiT belongs to generate-video, which declares it.
  // Counting it here as well would charge the graph for it twice --
  // this plan dedups by nothing, since it deliberately knows bytes and
  // not checkpoints.
  const bool destroys = _unload_cfg == model_memory::UnloadPolicy::kDestroy;
  const bool reclaims  = _unload_cfg == model_memory::UnloadPolicy::kAuto;
  // The encoder streams its layers here (ecfg.lm.stream_layers), so it
  // has a real floor; the VAE has none and counts at its size.
  m.hold(enc, model_memory::dir_weights_bytes(enc),
         genai::MiniMaxH3TextEncoder::streaming_floor_bytes(enc),
         destroys, reclaims);
  m.hold(vvae, model_memory::dir_weights_bytes(vvae), 0, destroys, reclaims);
  return m;
}

std::vector<ResourceClaim>
VideoRefEncoderStage::declare_resources() const
{
  if (_hf_dir.empty()) { return {}; }
  namespace fs = std::filesystem;
  const std::string root = resolve_model_dir(session(), _hf_dir);
  std::string enc = genai::MiniMaxH3TextEncoder::resolve_encoder_dir(root);
  if (enc.empty()) { enc = (fs::path(root) / "text_encoder").string(); }
  std::string vvae = genai::MetalMiniMaxH3VideoVae::resolve_vae_dir(root);
  if (vvae.empty()) { vvae = (fs::path(root) / "video_vae").string(); }
  // The DiT is declared here as well, exactly as the conditioner
  // declares the DiT it is paired with: this claim is what every peer
  // sizes itself against, and a 66 GB component nobody declared is the
  // silent under-count the resource-planning phase exists to prevent.
  // Which partition the reference names, from the models DB: a repo
  // holding both resolves to fl2va by filename, and this claim is what
  // every peer sizes itself against.
  const std::string part =
      genai::MetalMiniMaxH3Transformer::partition_of_model_type(
          resolve_model(session(), _hf_dir).model_type);
  std::string dit =
      genai::MetalMiniMaxH3Transformer::resolve_dit_dir(root, part);
  if (dit.empty()) { dit = (fs::path(root) / "transformer").string(); }
  return model_memory::weight_claims({enc, vvae, dit});
}

Job
VideoRefEncoderStage::initialize(RuntimeContext& ctx)
{
  // If a previous run left the weights UNLOADED (the idle-unload policy
  // drops them between requests), let this launch load them again: the
  // once-only guard is per-Stage, not per-launch.
  if (!_enc && !_video_vae) {
    _load_attempted = false;
    _unloaded       = false;
  }
  const bool model_from_iport =
      ctx.num_iports() > kModelPort && ctx.iport_connected(kModelPort);
  if (!model_from_iport) { ensure_loaded_(); }
  co_return;
}

void
VideoRefEncoderStage::ensure_loaded_()
{
  if (_load_attempted) { return; }
  _load_attempted = true;
  if (_hf_dir.empty()) {
    session()->error(fmt(
        "VideoRefEncoderStage('{}'): no model -- set config.hf_dir or wire a "
        "model-select source to the model iport; inert", this->id()));
    return;
  }
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr) {
    session()->error(fmt(
        "VideoRefEncoderStage('{}'): no metal-compute backend; inert",
        this->id()));
    return;
  }
  {
    const ResolvedModel rm = resolve_model(session(), _hf_dir);
    _root = rm.dir;
    _partition = genai::MetalMiniMaxH3Transformer::partition_of_model_type(
        rm.model_type);
  }
  if (!load_models_(mc)) {
    session()->error(fmt(
        "VideoRefEncoderStage('{}'): reference encoders failed to load; "
        "inert", this->id()));
  }
}

bool
VideoRefEncoderStage::load_models_(metal_compute::MetalCompute* mc)
{
  // ---- the conditioner + its vision tower ---------------------------
  _enc_dir = genai::MiniMaxH3TextEncoder::resolve_encoder_dir(_root);
  if (_enc_dir.empty()) { _enc_dir = _root; }
  genai::MiniMaxH3TextEncoder::Config ecfg;
  std::string eerr;
  if (!genai::MiniMaxH3TextEncoder::config_from_json(_enc_dir, ecfg, &eerr)) {
    session()->error(fmt("VideoRefEncoderStage('{}'): {}", this->id(), eerr));
    return false;
  }
  // A ref2va presentation is not a prompt-sized sequence: every image
  // reference is read at its own 2048 short edge and contributes
  // thousands of vision tokens, so the text path's 4096 pool overflows
  // on the FIRST reference. Sized from config rather than raised
  // globally, because the pool is KV for 50 layers of a 32B model and a
  // text-only conditioner should not pay for it.
  if (_max_prompt_tokens > 0) { ecfg.lm.max_seq = _max_prompt_tokens; }
  {
    // The 50 tapped layers are ~48 GB bf16 (~26 GB at w8), and this
    // stage holds two VAEs beside them. Same streaming rule the DiTs
    // and the conditioner use, sized on the COMPONENT rather than on
    // whatever the config named -- a repack root holds the 66 GB DiT
    // next to the encoder, and summing the tree would decide to stream
    // on the DiT's bytes.
    const auto plan = model_memory::plan_streaming(
        session(), _enc_dir, std::string(), model_memory::kStreamHeadroom);
    ecfg.lm.stream_layers = plan.stream;
    ecfg.lm.pin_frac      = plan.pin_frac;
    if (const char* e = std::getenv("VPIPE_H3_ENC_STREAM")) {
      ecfg.lm.stream_layers = (std::atoi(e) != 0);
      if (!ecfg.lm.stream_layers) { ecfg.lm.pin_frac = 0.0; }
    }
    session()->log_debug(fmt(
        "VideoRefEncoderStage('{}'): conditioner footprint {} GB (others {} "
        "GB) -> {}", this->id(), plan.footprint >> 30, plan.others >> 30,
        ecfg.lm.stream_layers ? "STREAM layers" : "PRELOAD"));
  }
  _enc = genai::MiniMaxH3TextEncoder::load(
      _enc_dir, mc, const_cast<SessionContextIntf*>(session()), ecfg);
  if (!_enc) {
    session()->error(fmt(
        "VideoRefEncoderStage('{}'): MiniMax-H3 conditioner load failed: {}",
        this->id(), _enc_dir));
    return false;
  }
  std::string verr;
  _vision = genai::MiniMaxH3TextEncoder::load_vision(_enc_dir, mc, ecfg,
                                                     &verr);
  if (!_vision) {
    // Fatal here, unlike on the text-only path: without a tower every
    // reference's vision block is missing and the presentation
    // describes pictures the model cannot see.
    session()->error(fmt(
        "VideoRefEncoderStage('{}'): the vision tower failed to load ({}); "
        "a ref2va request cannot be encoded without it", this->id(),
        verr.empty() ? "unknown error" : verr));
    return false;
  }

  // ---- the two VAEs, encoder halves ---------------------------------
  const std::string vdir =
      genai::MetalMiniMaxH3VideoVae::resolve_vae_dir(_root);
  genai::MetalMiniMaxH3VideoVae::Config vcfg;
  std::string vcerr;
  if (!genai::MetalMiniMaxH3VideoVae::config_from_json(
          vdir.empty() ? _root : vdir, vcfg, &vcerr)) {
    session()->error(fmt("VideoRefEncoderStage('{}'): {}", this->id(),
                         vcerr));
    return false;
  }
  _video_vae = genai::MetalMiniMaxH3VideoVae::load(
      vdir.empty() ? _root : vdir, mc, vcfg);
  if (!_video_vae) {
    session()->error(fmt(
        "VideoRefEncoderStage('{}'): video VAE load failed: {}", this->id(),
        vdir));
    return false;
  }

  const std::string adir =
      genai::MetalMiniMaxH3AudioVae::resolve_vae_dir(_root);
  genai::MetalMiniMaxH3AudioVae::Config acfg;
  std::string acerr;
  if (genai::MetalMiniMaxH3AudioVae::config_from_json(
          adir.empty() ? _root : adir, acfg, &acerr)) {
    _audio_vae = genai::MetalMiniMaxH3AudioVae::load(
        adir.empty() ? _root : adir, mc, acfg);
  }
  if (!_audio_vae) {
    // NOT fatal: a request of images and silent clips never touches it.
    // A soundtrack that then arrives is refused by name rather than
    // dropped, which is what encode_references does.
    session()->warn(fmt(
        "VideoRefEncoderStage('{}'): no audio VAE ({}); references that "
        "carry a soundtrack will be refused", this->id(),
        acerr.empty() ? "load failed" : acerr));
  }
  // What the idle-unload heuristic weighs: this stage's own components
  // plus the DiT it is paired with. Same dirs declare_resources() named,
  // so the declaration and the estimate cannot disagree.
  _peer_dirs.clear();
  _peer_dirs.push_back(_enc_dir);
  if (!vdir.empty()) { _peer_dirs.push_back(vdir); }
  if (!adir.empty()) { _peer_dirs.push_back(adir); }
  {
    const std::string d =
        genai::MetalMiniMaxH3Transformer::resolve_dit_dir(_root, _partition);
    if (!d.empty()) { _peer_dirs.push_back(d); }
  }
  session()->info(fmt(
      "VideoRefEncoderStage('{}'): MiniMax-H3 ref2va encoders ready "
      "(conditioner tap {}{}, video VAE, {})", this->id(), ecfg.tap,
      _enc->streaming() ? " streamed" : "",
      _audio_vae ? "audio VAE" : "NO audio VAE"));
  return true;
}

// ---- the request ----------------------------------------------------

bool
VideoRefEncoderStage::decode_references_(std::vector<h3::MediaReference>* out)
{
  const FFmpegLibraries* libs = session()->services()->ffmpeg_libraries();
  if (libs == nullptr || !libs->valid()) {
    session()->warn(fmt(
        "VideoRefEncoderStage('{}'): FFmpeg is unavailable, so no reference "
        "media can be decoded", this->id()));
    return false;
  }
  const int audio_rate =
      _audio_vae ? _audio_vae->config().sample_rate : 32000;
  // Every reference is truncated to the GENERATED duration, so nothing
  // past it is ever decoded. On a long source that is the difference
  // between reading five seconds and reading the file.
  const int aligned = h3::align_num_frames(_frames, 17, 5);
  const double max_seconds = (double)aligned / h3::kFps;

  for (std::size_t i = 0; i < _references.size(); ++i) {
    const std::string& path = _references[i];
    std::string derr;
    // WHAT IT IS, from the file. One header read, no frame decoded --
    // and the answer decides which decoder runs, which matters because
    // the wrong one does not fail: an image read as video comes back as
    // a one-frame clip, and a clip read as an image comes back as its
    // first frame.
    auto probe = probe_media_file(libs, path, &derr);
    if (!probe) {
      session()->warn(fmt(
          "VideoRefEncoderStage('{}'): reference {} ('{}') will not open: {}",
          this->id(), i + 1, path, derr));
      return false;
    }
    if (probe->kind == MediaKind::Unknown) {
      session()->warn(fmt(
          "VideoRefEncoderStage('{}'): reference {} ('{}') carries neither "
          "video nor audio", this->id(), i + 1, path));
      return false;
    }
    session()->log_debug(fmt(
        "VideoRefEncoderStage('{}'): reference {} ('{}') is {} ({}x{}"
        "{})", this->id(), i + 1, path, media_kind_name(probe->kind),
        probe->width, probe->height,
        probe->kind == MediaKind::Video
            ? fmt(", {:.2f} fps, {:.2f} s", probe->fps,
                  probe->duration_seconds)()
            : std::string()));

    h3::MediaReference m;
    if (probe->kind == MediaKind::Audio) {
      auto a = decode_audio_file(libs, path, audio_rate, &derr, 2);
      if (!a) {
        session()->warn(fmt(
            "VideoRefEncoderStage('{}'): reference {} ('{}') would not decode "
            "as audio: {}", this->id(), i + 1, path, derr));
        return false;
      }
      m.kind        = h3::MediaReference::Kind::kAudio;
      m.channels    = a->channels;
      m.sample_rate = a->sample_rate;
      m.pcm         = std::move(a->pcm);
      out->push_back(std::move(m));
      continue;
    }
    if (probe->kind == MediaKind::Image) {
      auto im = decode_image_file(libs, path, &derr);
      if (!im) {
        session()->warn(fmt(
            "VideoRefEncoderStage('{}'): reference {} ('{}') would not decode "
            "as an image: {}", this->id(), i + 1, path, derr));
        return false;
      }
      m.kind       = h3::MediaReference::Kind::kImage;
      m.num_frames = 1;
      m.height     = im->height;
      m.width      = im->width;
      m.rgb        = std::move(im->rgb);
      out->push_back(std::move(m));
      continue;
    }

    // A video reference conditions on its own soundtrack when it has
    // one: the two came out of ONE open container, which is the only
    // way their clocks stay in sync.
    auto v = decode_video_file(libs, path, max_seconds,
                               probe->has_audio ? audio_rate : 0, &derr);
    if (!v) {
      session()->warn(fmt(
          "VideoRefEncoderStage('{}'): reference {} ('{}') would not decode "
          "as video: {}", this->id(), i + 1, path, derr));
      return false;
    }
    // The probe said video, but the truncation above can leave ONE
    // frame -- a 1/24 s `frames` on a long clip. One frame is a still by
    // every rule that follows (no rate to condition on, no motion to
    // read), so it is demoted rather than encoded as a clip of length 1.
    m.kind = v->num_frames > 1 ? h3::MediaReference::Kind::kVideo
                               : h3::MediaReference::Kind::kImage;
    m.num_frames  = v->num_frames;
    m.height      = v->height;
    m.width       = v->width;
    m.fps         = v->fps;
    m.rgb         = std::move(v->rgb);
    m.channels    = v->audio_channels;
    m.sample_rate = v->audio_sample_rate;
    m.pcm         = std::move(v->pcm);
    if (m.kind == h3::MediaReference::Kind::kImage) {
      // A still has no rate to condition on and no soundtrack.
      m.num_frames = 1;
      m.fps        = h3::kFps;
      m.channels   = 0;
      m.pcm.clear();
      session()->log_debug(fmt(
          "VideoRefEncoderStage('{}'): reference {} truncated to a single "
          "frame at {} frames; conditioning on it as a still", this->id(),
          i + 1, aligned));
    }
    out->push_back(std::move(m));
  }
  return true;
}


// ---- idle unload ----------------------------------------------------

void
VideoRefEncoderStage::resolve_unload_policy_()
{
  if (_unload_resolved) { return; }
  _unload_resolved = true;
  // Decided AFTER the init barrier, where every peer has loaded and
  // real bytes are authoritative -- the whole point of the barrier.
  switch (_unload_cfg) {
    case model_memory::UnloadPolicy::kAlways: _unload_idle = true;  break;
    case model_memory::UnloadPolicy::kNever:  _unload_idle = false; break;
    default:
      _unload_idle =
          model_memory::bounded(session(), _peer_dirs,
                                model_memory::kHeadroom);
      break;
  }
  session()->log_debug(fmt(
      "VideoRefEncoderStage('{}'): encoders + DiT footprint {} MB + {} MB "
      "headroom vs {} MB RAM, unload_when_idle={} -> {}", this->id(),
      model_memory::weight_footprint(session(), _peer_dirs) >> 20,
      model_memory::kHeadroom >> 20, model_memory::phys_ram() >> 20,
      model_memory::unload_policy_name(_unload_cfg),
      _unload_idle ? "UNLOAD after each request" : "keep resident"));
  if (_unload_idle) {
    session()->info(fmt(
        "VideoRefEncoderStage('{}'): memory-bounded -- the conditioner and "
        "the VAEs are dropped once a request is encoded and reloaded for the "
        "next one", this->id()));
  }
}

void
VideoRefEncoderStage::unload_models_()
{
  if (_unloaded) { return; }
  _enc.reset();
  _vision.reset();
  _video_vae.reset();
  _audio_vae.reset();
  _unloaded = true;
}

bool
VideoRefEncoderStage::reload_models_()
{
  if (!_unloaded) { return _enc != nullptr; }
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr) { return false; }
  _unloaded = false;
  return load_models_(mc);
}

// ---- process --------------------------------------------------------

Job
VideoRefEncoderStage::process(RuntimeContext& ctx)
{
  if (!_model_latched && ctx.num_iports() > kModelPort &&
      ctx.iport_connected(kModelPort)) {
    auto mb = co_await ctx.read(kModelPort);
    _model_latched = true;
    if (const auto* mfd =
            mb ? dynamic_cast<const FlexDataPayload*>(mb.get()) : nullptr) {
      apply_model_select_beat(mfd->data, _hf_dir);
    }
    ensure_loaded_();
  }

  auto pb = co_await ctx.read(0);
  if (!pb) { ctx.signal_done(); co_return; }
  const auto* pfd = dynamic_cast<const FlexDataPayload*>(pb.get());
  if (pfd == nullptr) {
    session()->warn(fmt(
        "VideoRefEncoderStage('{}'): expected a FlexData prompt, got {}; "
        "skipping", this->id(), pb->describe()));
    co_return;
  }
  const std::string prompt = prompt_of_(pfd->data);

  resolve_unload_policy_();
  if (_unloaded && !reload_models_()) {
    session()->warn(fmt(
        "VideoRefEncoderStage('{}'): the reference encoders did not reload; "
        "skipping", this->id()));
    co_return;
  }
  if (!_enc || !_vision || !_video_vae) {
    session()->warn(fmt(
        "VideoRefEncoderStage('{}'): no reference encoders; skipping",
        this->id()));
    co_return;
  }

  std::vector<h3::MediaReference> refs;
  if (!decode_references_(&refs)) { co_return; }
  std::string verr;
  if (!h3::validate_reference_request(refs, _limits, &verr)) {
    session()->warn(fmt("VideoRefEncoderStage('{}'): {}; skipping",
                        this->id(), verr));
    co_return;
  }

  h3::ReferencePlan plan;
  plan.target_frames = h3::align_num_frames(_frames, 17, 5);
  plan.reference_image_short_edge = _ref_short_edge;
  plan.video_sample_fps = _video_sample_fps;
  {
    // The DiT's patch, from the checkpoint rather than assumed: the
    // rows this stage packs are read by a transformer that reshapes
    // them, so a patch mismatch is not a fidelity question.
    genai::MetalMiniMaxH3Transformer::Config dcfg;
    if (genai::MetalMiniMaxH3Transformer::config_from_json(_root, dcfg,
                                                           nullptr,
                                                           _partition)) {
      plan.patch_h = dcfg.patch_h;
      plan.patch_w = dcfg.patch_w;
    }
  }

  h3::ReferenceEncoders models;
  models.mc        = session()->services()->metal_compute();
  models.vision    = _vision.get();
  models.video_vae = _video_vae.get();
  models.audio_vae = _audio_vae.get();
  models.text      = _enc.get();
  {
    auto* ui = session();
    const std::string label = "encoding " + std::to_string(refs.size()) +
                              " reference(s)";
    models.progress = [ui, label](int done, int total) {
      ui->log_debug(fmt("{}: {}/{}", label, done, total));
    };
  }

  h3::EncodedReferences enc;
  std::string eerr;
  if (!h3::encode_references(refs, prompt, plan, models, &enc, &eerr)) {
    session()->warn(fmt("VideoRefEncoderStage('{}'): {}; skipping",
                        this->id(), eerr));
    co_return;
  }

  // ---- emit ---------------------------------------------------------
  // The reference rows go out FIRST: generate-video reads the
  // conditioning as its required beat and polls the rest, so having
  // them already queued is what keeps that poll from being a race.
  const int vrows = enc.video_row_elems > 0
                        ? (int)(enc.video_rows.size() /
                                (std::size_t)enc.video_row_elems)
                        : 0;
  {
    auto t = std::make_unique<TensorBeatPayload>();
    t->dtype = TensorBeat::DType::F32;
    t->shape = {vrows, std::max(enc.video_row_elems, 1)};
    t->resize_contiguous(enc.video_rows.size());
    if (!enc.video_rows.empty()) {
      std::memcpy(t->as_f32(), enc.video_rows.data(),
                  enc.video_rows.size() * sizeof(float));
    }
    co_await ctx.write(1, std::move(t));
  }
  const int arows = enc.audio_row_elems > 0
                        ? (int)(enc.audio_rows.size() /
                                (std::size_t)enc.audio_row_elems)
                        : 0;
  {
    auto t = std::make_unique<TensorBeatPayload>();
    t->dtype = TensorBeat::DType::F32;
    t->shape = {arows, std::max(enc.audio_row_elems, 1)};
    t->resize_contiguous(enc.audio_rows.size());
    if (!enc.audio_rows.empty()) {
      std::memcpy(t->as_f32(), enc.audio_rows.data(),
                  enc.audio_rows.size() * sizeof(float));
    }
    co_await ctx.write(2, std::move(t));
  }

  {
    auto t = std::make_unique<TensorBeatPayload>();
    t->dtype = TensorBeat::DType::Bf16;
    t->shape = {enc.n_tokens, _enc->config().text_dim};
    const std::size_t n =
        (std::size_t)enc.n_tokens * _enc->config().text_dim;
    t->resize_contiguous(n);
    std::memcpy(t->data.data(), enc.conditioning.contents(), n * 2);

    // The PLAN travels with the conditioning, because the two are one
    // request: the tags say which rows are a vision block (MiniMax-H3
    // tags those as video, not text) and the reference geometry is what
    // the packed layout reserves rows for. Splitting them across beats
    // would let a graph pair a conditioning with another request's
    // layout, which packs and then fails 50 layers deep.
    FlexData sb = FlexData::make_object();
    auto o = sb.as_object();
    o.insert_or_assign("partition", FlexData::make_string("ref2va"));
    FlexData tags = FlexData::make_array();
    for (int v : enc.token_tags) {
      tags.as_array().push_back(FlexData::make_int(v));
    }
    o.insert_or_assign("token_tags", std::move(tags));
    FlexData rl = FlexData::make_array();
    for (const auto& L : enc.layout) {
      FlexData r = FlexData::make_object();
      auto ro = r.as_object();
      ro.insert_or_assign(
          "kind",
          FlexData::make_string(
              L.kind == h3::Reference::Kind::kImage   ? "image"
              : L.kind == h3::Reference::Kind::kVideo ? "video"
                                                      : "audio"));
      ro.insert_or_assign("latent_frames",
                          FlexData::make_int(L.num_latent_frames));
      ro.insert_or_assign("latent_height",
                          FlexData::make_int(L.latent_height));
      ro.insert_or_assign("latent_width",
                          FlexData::make_int(L.latent_width));
      ro.insert_or_assign("audio_latents",
                          FlexData::make_int(L.num_audio_latents));
      rl.as_array().push_back(std::move(r));
    }
    o.insert_or_assign("references", std::move(rl));
    t->sideband = std::move(sb);
    co_await ctx.write(0, std::move(t));
  }

  session()->info(fmt(
      "VideoRefEncoderStage('{}'): {} reference(s) -> {} conditioning rows, "
      "{} reference video rows, {} reference audio rows", this->id(),
      refs.size(), enc.n_tokens, vrows, arows));
  ++_emitted;
  if (_unload_idle) { unload_models_(); }
}

#endif  // VPIPE_BUILD_APPLE_SILICON

VPIPE_REGISTER_STAGE(VideoRefEncoderStage)
VPIPE_REGISTER_SPEC(VideoRefEncoderStage, kSpec)

}  // namespace vpipe
