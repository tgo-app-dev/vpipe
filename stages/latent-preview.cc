#include "stages/latent-preview.h"

#include "common/vpipe-format.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/tensor-beat.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "pipeline/runtime-context.h"
#include "pipeline/resource-plan.h"
#include "stages/model-memory.h"
#include "stages/model-registry.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <utility>

namespace vpipe {

bool
LatentPreviewSpec::due(int step, int total) const noexcept
{
  if (!enabled() || step <= 0) { return false; }
  return step >= total || step % every == 0;
}

LatentPreviewSpec
LatentPreviewSpec::from_model_config(const FlexData& cfg)
{
  LatentPreviewSpec s;
  if (!cfg.is_object()) { return s; }
  auto o = cfg.as_object();
  if (o.contains(latent_preview::kVaeKey)) {
    s.vae = std::string(o.at(latent_preview::kVaeKey).as_string(""));
  }
  if (o.contains(latent_preview::kEveryKey)) {
    s.every = (int)o.at(latent_preview::kEveryKey).as_int(s.every);
  }
  if (o.contains(latent_preview::kMaxEdgeKey)) {
    s.max_edge = (int)o.at(latent_preview::kMaxEdgeKey).as_int(s.max_edge);
  }
  if (o.contains(latent_preview::kFramesKey)) {
    s.max_frames =
        (int)o.at(latent_preview::kFramesKey).as_int(s.max_frames);
  }
  s.every = std::max(0, s.every);
  s.max_edge = std::max(0, s.max_edge);
  s.max_frames = std::max(0, s.max_frames);
  return s;
}

void
LatentPreviewSpec::emit(FlexData& cfg) const
{
  if (vae.empty() || !cfg.is_object()) { return; }
  auto o = cfg.as_object();
  o.insert_or_assign(latent_preview::kVaeKey, FlexData::make_string(vae));
  o.insert_or_assign(latent_preview::kEveryKey, FlexData::make_int(every));
  o.insert_or_assign(latent_preview::kMaxEdgeKey,
                     FlexData::make_int(max_edge));
  o.insert_or_assign(latent_preview::kFramesKey,
                     FlexData::make_int(max_frames));
}

int
latent_preview_shrink(int w, int h, int max_edge)
{
  const int edge = std::max(w, h);
  if (max_edge <= 0 || edge <= max_edge) { return 1; }
  return (edge + max_edge - 1) / max_edge;
}

#ifdef VPIPE_BUILD_APPLE_SILICON

namespace {

// The TAE a `preview_vae` names: a direct file, a registered model's
// pinned checkpoint, or a directory with one .safetensors in it. Wider
// than resolve_adapter_file() on purpose, and only here: madebyollin
// publishes some TAEs as torch `.pth` only (taeqi2_1), which the
// checkpoint reader maps like any shard, while an ADAPTER field
// accepting .pth would change what every LoRA picker resolves to.
std::string
resolve_tae_file_(const SessionContextIntf* session, const std::string& ref,
                  std::string* err)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  if (!ref.empty() && fs::is_regular_file(fs::path(ref), ec)) { return ref; }
  const ResolvedModel rm = resolve_model(session, ref);
  if (rm.from_registry) {
    for (const std::string& f : rm.files) {
      const fs::path p = fs::path(rm.dir) / f;
      const std::string ext = p.extension().string();
      if ((ext == ".safetensors" || ext == ".pth") &&
          fs::is_regular_file(p, ec)) {
        return p.string();
      }
    }
  }
  return resolve_adapter_file(session, ref, err);
}

}  // namespace

std::vector<ResourceClaim>
latent_preview_claims(const SessionContextIntf* session,
                      std::string_view owner, const LatentPreviewSpec& spec,
                      genai::MetalTaeDecoder::Trim trim, int w, int h,
                      int frames)
{
  if (!spec.enabled() || w <= 0 || h <= 0 || frames <= 0) { return {}; }
  std::string err;
  const std::string file = resolve_tae_file_(session, spec.vae, &err);
  if (file.empty()) { return {}; }
  std::vector<ResourceClaim> out = model_memory::weight_claims({file});
  using Tae = genai::MetalTaeDecoder;
  Tae::Plan tp;
  if (!Tae::plan(file, 1, 1, 1, trim, &tp, &err) || tp.info.spatial <= 0) {
    return out;
  }
  int T = 1;
  while (Tae::frames_for(T, tp.info.time_upscale, trim) < frames &&
         T < frames) {
    ++T;
  }
  const int lh = h / tp.info.spatial, lw = w / tp.info.spatial;
  if (lh <= 0 || lw <= 0 || !Tae::plan(file, T, lh, lw, trim, &tp, &err)) {
    return out;
  }
  // Plus the host's frames: one native frame widened to f32 for the box
  // filter, and the shrunk output twice -- the render's buffer and the
  // beat in flight.
  const int NH = lh * tp.info.spatial, NW = lw * tp.info.spatial;
  const int k = latent_preview_shrink(NW, NH, spec.max_edge);
  int nf = Tae::frames_for(T, tp.info.time_upscale, trim);
  if (spec.max_frames > 0) { nf = std::min(nf, spec.max_frames); }
  const std::size_t ic = (std::size_t)tp.info.image_channels;
  // THE CHECKPOINT IS NOT WHAT THE DECODER HOLDS, and the claim above
  // books the checkpoint: the weights planner sizes a weight claim from
  // the file on disk. MetalTaeDecoder narrows to f16 on the way in
  // (which makes an f32 TAESD file an over-estimate, harmless) and, on
  // matrix-core hardware, keeps an hwio twin of every 3x3 for the
  // hardware convolution -- which makes taeh3 39.3 MB held against 22.7
  // MB on disk. The file was a fair proxy until that twin existed; it
  // now under-books by ~16 MB on three of the five published TAEs, and
  // a claim that under-books is the direction that admits a graph which
  // does not fit. Top it up from plan(), the same sizing the decoder
  // allocates from, with NO phase -- the decoder is held from the first
  // preview to the end of the run, exactly as the checkpoint claim
  // above is.
  const std::size_t on_disk = model_memory::dir_weights_bytes(file);
  if (tp.weight_bytes > on_disk) {
    for (auto& c : model_memory::scratch_claims(
             fmt("{}-preview-weights", owner)(), tp.weight_bytes - on_disk,
             /*phase=*/{})) {
      out.push_back(std::move(c));
    }
  }
  const std::size_t bytes =
      tp.scratch_bytes + (k > 1 ? 4 * ic * NW * NH : 0) +
      2 * (std::size_t)nf * ic * (std::size_t)(NW / k) * (std::size_t)(NH / k);
  for (auto& c : model_memory::scratch_claims(fmt("{}-preview", owner)(),
                                              bytes,
                                              model_memory::kPhaseDenoise)) {
    out.push_back(std::move(c));
  }
  return out;
}

LatentPreviewer::LatentPreviewer(const SessionContextIntf* session,
                                 std::string owner)
  : _session(session), _owner(std::move(owner))
{
}

LatentPreviewer::~LatentPreviewer()
{
  // A stage torn down mid-generation: nothing may outlive the context
  // the worker writes through.
  _abandon = true;
  {
    std::lock_guard<std::mutex> g(_mu);
    _finishing = true;
  }
  _cv.notify_all();
  if (_worker.joinable()) { _worker.join(); }
}

void
LatentPreviewer::configure(const LatentPreviewSpec& spec,
                           genai::MetalTaeDecoder::Trim trim)
{
  // The decoder is swapped lazily in begin(): this can be called between
  // generations only (it is the stage's model_config latch), but loading
  // there keeps a spec nobody previews from ever touching the disk.
  if (spec.vae != _spec.vae) { _failed_vae.clear(); }
  _spec = spec;
  _trim = trim;
}

bool
LatentPreviewer::load_(std::string* err)
{
  std::string file = resolve_tae_file_(_session, _spec.vae, err);
  if (file.empty()) { return false; }
  auto* mc = _session->services()->metal_compute();
  if (mc == nullptr) {
    *err = "no metal-compute runtime";
    return false;
  }
  // Through the manager, like every checkpoint: it is small, but the
  // manager is the only thing with a session-wide view of what is held.
  auto ws = genai::open_weight_set(file, _session);
  if (ws == nullptr) {
    *err = fmt("no readable checkpoint at '{}'", file)();
    return false;
  }
  _dec = genai::MetalTaeDecoder::load(std::move(ws), mc, err);
  if (_dec == nullptr) { return false; }
  _loaded_vae = _spec.vae;
  const auto& in = _dec->info();
  _session->info(fmt(
      "LatentPreviewer('{}'): {} TAE '{}' -- {} latent channels, {}x "
      "spatial, {}x temporal, {:.1f}M params", _owner, in.layout, file,
      in.latent_channels, in.spatial, in.time_upscale,
      (double)in.params / 1e6));
  return true;
}

bool
LatentPreviewer::begin(RuntimeContext& ctx, unsigned port, double fps,
                       int video_frames)
{
  if (active()) { end(ctx); }
  if (!_spec.enabled()) { return false; }
  // Unwired is the common case and costs nothing: no load, no thread.
  if (port >= ctx.num_oports() || !ctx.has_consumers(port)) { return false; }
  if (_dec != nullptr && _loaded_vae != _spec.vae) { release(); }
  if (_dec == nullptr) {
    if (!_failed_vae.empty() && _failed_vae == _spec.vae) { return false; }
    std::string err;
    if (!load_(&err)) {
      _failed_vae = _spec.vae;
      _session->warn(fmt(
          "LatentPreviewer('{}'): no previews -- the preview VAE '{}' did "
          "not load: {}", _owner, _spec.vae, err));
      return false;
    }
  }
  _ctx = &ctx;
  _port = port;
  _fps = fps;
  _video_frames = video_frames;
  _have_job = false;
  _finishing = false;
  _abandon = false;
  _port_closed = false;
  _rendered = 0;
  _replaced = 0;
  _worker = std::thread([this] { worker_(); });
  return true;
}

bool
LatentPreviewer::due(int step, int total) const noexcept
{
  return active() && !_port_closed && _spec.due(step, total);
}

void
LatentPreviewer::submit(std::vector<float> latent, int C, int T, int h,
                        int w, int step, int total)
{
  if (!active()) { return; }
  {
    std::lock_guard<std::mutex> g(_mu);
    // LATEST WINS. A render still queued when the next arrives is stale
    // by definition, and queueing both would make the preview lag the
    // generation by however far the GPU has fallen behind.
    if (_have_job) { ++_replaced; }
    _job.latent = std::move(latent);
    _job.C = C;
    _job.T = T;
    _job.h = h;
    _job.w = w;
    _job.step = step;
    _job.total = total;
    _have_job = true;
  }
  _cv.notify_one();
}

void
LatentPreviewer::end(const RuntimeContext& ctx)
{
  if (!active()) { return; }
  // A stop abandons what is queued and cuts the decode in flight short;
  // an ordinary end lets them finish -- the last step's preview is the
  // one a viewer waits on while the real decode runs.
  if (ctx.stop_requested()) { _abandon = true; }
  {
    std::lock_guard<std::mutex> g(_mu);
    _finishing = true;
  }
  _cv.notify_all();
  _worker.join();
  _ctx = nullptr;
  if (_rendered.load() > 0 || _replaced.load() > 0) {
    _session->log_debug(fmt(
        "LatentPreviewer('{}'): {} previews rendered, {} replaced before "
        "they started", _owner, _rendered.load(), _replaced.load()));
  }
}

void
LatentPreviewer::release()
{
  if (active()) { return; }
  _dec.reset();
  _loaded_vae.clear();
}

std::size_t
LatentPreviewer::resident_bytes() const noexcept
{
  return _dec != nullptr ? _dec->resident_bytes() : 0;
}

void
LatentPreviewer::worker_()
{
  for (;;) {
    Job job;
    {
      std::unique_lock<std::mutex> l(_mu);
      _cv.wait(l, [&] { return _have_job || _finishing || _abandon; });
      if (_abandon.load() || !_have_job) { return; }
      job = std::move(_job);
      _have_job = false;
    }
    if (!_port_closed) { render_(job); }
  }
}

void
LatentPreviewer::render_(Job& job)
{
  const auto& in = _dec->info();
  const int shrink = latent_preview_shrink(job.w * in.spatial,
                                           job.h * in.spatial,
                                           _spec.max_edge);
  genai::MetalTaeDecoder::Frames f;
  std::string err;
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = _dec->decode(
      job.latent.data(), job.C, job.T, job.h, job.w, _trim,
      _video_frames > 0 ? _spec.max_frames : 1, shrink, &f, &err,
      [this] { return _abandon.load(); });
  if (!ok) {
    if (!err.empty()) {
      _session->warn(fmt("LatentPreviewer('{}'): step {} preview: {}",
                         _owner, job.step, err));
    }
    return;
  }
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
  if (f.frames <= 0) { return; }

  // RGB, or RGBA with straight alpha from an RGBA TAE -- which a preview
  // stage composites over a checkerboard, as it does the real decode's.
  auto beat = std::make_unique<TensorBeatPayload>();
  beat->dtype = TensorBeat::DType::U8;
  if (_video_frames > 0) {
    beat->shape = {f.frames, f.channels, f.height, f.width};
  } else {
    beat->shape = {f.channels, f.height, f.width};
  }
  const std::size_t n = (std::size_t)f.channels * f.height * f.width *
                        (_video_frames > 0 ? f.frames : 1);
  beat->resize_contiguous(n);
  std::memcpy(beat->as_u8(), f.rgb.data(), n);

  FlexData sb = FlexData::make_object();
  auto o = sb.as_object();
  o.insert_or_assign("preview", FlexData::make_bool(true));
  o.insert_or_assign("step", FlexData::make_int(job.step));
  o.insert_or_assign("steps", FlexData::make_int(job.total));
  if (_video_frames > 0) {
    // The rate that plays THIS clip over the real one's duration: a TAE
    // that makes fewer frames than the VAE (a 2D one makes one per latent
    // frame) plays them slower rather than the clip playing short.
    double fps = _fps;
    const int full = _dec->frames_for(job.T, _trim);
    if (full > 0 && full != _video_frames) {
      fps = _fps * (double)full / (double)_video_frames;
    }
    const std::int64_t num = std::llround(fps * 1000.0);
    o.insert_or_assign("frames", FlexData::make_int(f.frames));
    o.insert_or_assign("fps", FlexData::make_real(fps));
    if (num > 0) {
      o.insert_or_assign("fps_num", FlexData::make_uint((std::uint64_t)num));
      o.insert_or_assign("fps_den", FlexData::make_uint(1000));
    }
  }
  beat->sideband = std::move(sb);
  // Diagnostic: every render as raw planar U8, so a preview can be looked
  // at without a browser attached. The name carries the shape.
  if (const char* dd = std::getenv("VPIPE_PREVIEW_DUMP")) {
    const std::string path =
        fmt("{}/{}-step{:03d}-{}x{}x{}x{}.u8", dd, _owner, job.step,
            _video_frames > 0 ? f.frames : 1, f.channels, f.height,
            f.width)();
    if (std::FILE* fp = std::fopen(path.c_str(), "wb")) {
      std::fwrite(beat->as_u8(), 1, n, fp);
      std::fclose(fp);
    }
  }
  if (!_ctx->write_sync(_port, std::move(beat))) {
    _port_closed = true;   // teardown: stop rendering for nobody
    return;
  }
  ++_rendered;
  _session->log_debug(fmt(
      "LatentPreviewer('{}'): step {}/{} preview, {} frame(s) at {}x{} in "
      "{:.0f} ms", _owner, job.step, job.total, f.frames, f.width, f.height,
      ms));
}

#endif  // VPIPE_BUILD_APPLE_SILICON

}  // namespace vpipe
