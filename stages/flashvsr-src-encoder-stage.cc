#include "stages/flashvsr-src-encoder-stage.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "pipeline/stage-registry.h"
#include "pipeline/stage-spec.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "generative-models/flashvsr/flashvsr-family.h"
#include "generative-models/weight-set.h"
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

constexpr unsigned kSourcePort = 0;
constexpr unsigned kModelPort  = 1;

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "the FlashVSR model directory (its converted layout: the denoiser, "
          "the source projection and the fixed context as one checkpoint). A "
          "model-select source on iport1 overrides it, which is how the "
          "shipped graph shares one model choice across the encoder, the "
          "denoiser and the VAE"},
};
const PortSpec kIports[] = {
  {.name = "source",
   .doc = "the low-quality CLIP in pixels: planar U8 RGB [frames, 3, H, W] "
          "with `fps` on the sideband, which is the shape temporal-stack "
          "builds. REQUIRED -- this family conditions on its source and "
          "nothing else, so an unwired port is a refusal and not a fallback: "
          "a super-resolver handed no source would produce a confident "
          "upscale of noise",
   .type = &typeid(TensorBeatPayload),
   .tags = "video", .clock_group = 0},
  {.name = "model",
   .doc = "OPTIONAL shared model reference from a model-select source; "
          "overrides the hf_dir config",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "conditioning",
   .doc = "bf16 [row_frames * tokens, hidden]: the source projected onto the "
          "denoiser's own token grid, one row frame per FOUR source frames. "
          "This is ORDINARY conditioning -- it feeds generate-video's "
          "existing conditioning iport, which needed no change to accept it. "
          "The sideband carries the geometry the rows belong to "
          "{source_width, source_height, source_frames, row_frames}",
   .type = &typeid(TensorBeatPayload),
   .tags = "conditioning", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "flashvsr-src-encoder",
  .doc       = "FlashVSR's source encoder: a low-quality video clip in "
               "PIXELS through the model's learned causal projection, into "
               "the conditioning rows its denoiser adds to the residual "
               "stream. Model-specific -- this family has no text encoder at "
               "all, because its text context is a constant its checkpoint "
               "carries, and its conditioning is the clip itself. Feed "
               "generate-video's conditioning iport.",
  .display_name = "FlashVSR Source Encoder",
  .category  = StageCategory::ModelSpecificConfig,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

FlashVsrSrcEncoderStage::FlashVsrSrcEncoderStage(
    const SessionContextIntf* session, std::string id,
    std::vector<InEdge> iports, FlexData config)
    : TypedStage<FlashVsrSrcEncoderStage>(session, std::move(id),
                                          std::move(iports),
                                          std::move(config))
{
  // The spec DESCRIBES the oport; this is what CREATES it. Without it
  // the stage has zero oports, and every graph that wires its
  // conditioning onward -- the shipped flashvsr-restore one included --
  // is refused at pipeline_from_spec before anything loads.
  allocate_oports(spec().oports.size());
  // Deferred validation: constructors never throw here, so a bad config
  // is recorded and the runtime skips the stage at launch.
  _hf_dir = attr_str("hf_dir");
}

FlashVsrSrcEncoderStage::~FlashVsrSrcEncoderStage() = default;

const StageSpec&
FlashVsrSrcEncoderStage::spec() const noexcept
{
  return kSpec;
}

void
FlashVsrSrcEncoderStage::apply_constant(unsigned iport, const FlexData& beat)
{
  // The pre-launch twin of the runtime latch in process(): the same beat
  // and the same parse, early enough that declare_resources() sees the
  // model. Bookkeeping only -- nothing loads here.
  if (iport != kModelPort) { return; }
  apply_model_select_beat(beat, _hf_dir);
}

#ifdef VPIPE_BUILD_APPLE_SILICON

std::vector<ResourceClaim>
FlashVsrSrcEncoderStage::declare_resources() const
{
  const std::string root = resolve_model_dir(session(), _hf_dir);
  if (root.empty()) { return {}; }
  // The projection's OWN set, which is the name the manager keys its load
  // by: the root when converted, LQ_proj_in.ckpt as published.
  genai::FlashVsrLayout layout;
  if (!genai::resolve_flashvsr_layout(root, &layout)) {
    return model_memory::weight_claims({root});
  }
  return model_memory::weight_claims({layout.source});
}

StageMemory
FlashVsrSrcEncoderStage::declare_memory() const
{
  // The projection, under the set it loads through -- the root when
  // converted, where the denoiser's holding names it too and the two merge
  // (that holding's floor counts the projection), and LQ_proj_in.ckpt as
  // published. It does not stream, so its floor is what it weighs. Its
  // WORKING SET is booked by the denoiser's family, which knows the clip
  // geometry this stage only sees at its first beat.
  StageMemory m;
  const std::string root = resolve_model_dir(session(), _hf_dir);
  if (root.empty()) { return m; }
  genai::FlashVsrLayout layout;
  if (!genai::resolve_flashvsr_layout(root, &layout)) { return m; }
  const std::size_t b = genai::FlashVsrLqProj::weight_bytes(
      layout.source, layout.source_prefix);
  m.hold(layout.source, b, b);
  return m;
}

void
FlashVsrSrcEncoderStage::reset_run_state()
{
  // Let a relaunch load again if the weights are gone; keep the guard
  // set while they are still held, because reloading on top of a
  // resident copy is what doubles peak memory.
  if (_proj == nullptr) { _load_attempted = false; }
  _model_latched = false;
  _emitted = 0;
}

bool
FlashVsrSrcEncoderStage::ensure_loaded_()
{
  if (_proj != nullptr) { return true; }
  if (_load_attempted) { return false; }
  _load_attempted = true;
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr) { return false; }
  const std::string root = resolve_model_dir(session(), _hf_dir);
  if (root.empty()) {
    session()->error(fmt(
        "FlashVsrSrcEncoderStage('{}'): no model directory -- set hf_dir or "
        "wire a model-select source", this->id()));
    return false;
  }
  genai::FlashVsrLayout layout;
  if (!genai::resolve_flashvsr_layout(root, &layout)) {
    session()->error(fmt(
        "FlashVsrSrcEncoderStage('{}'): '{}' is not a FlashVSR checkpoint -- "
        "neither the converted layout nor the published files; inert",
        this->id(), root));
    return false;
  }
  std::string err;
  _proj = genai::FlashVsrLqProj::load(
      genai::open_weight_set(layout.source, session()), mc, &err,
      layout.source_prefix);
  if (_proj == nullptr) {
    session()->error(fmt(
        "FlashVsrSrcEncoderStage('{}'): source projection load failed for "
        "'{}': {}; inert", this->id(), root, err));
    return false;
  }
  session()->info(fmt(
      "FlashVsrSrcEncoderStage('{}'): source projection {} wide, {} output "
      "layer(s), {}x{} pixel unshuffle -- no prompt and no text encoder",
      this->id(), _proj->config().out_dim, _proj->config().n_out_layers,
      _proj->config().shuffle_w, _proj->config().shuffle_h));
  return true;
}

Job
FlashVsrSrcEncoderStage::initialize(RuntimeContext& ctx)
{
  // With a model-select wired the choice only lands after the init
  // barrier, so the load waits for the first beat. Without one, hf_dir
  // is all there is and loading now keeps the first clip from paying
  // for it.
  if (!(ctx.num_iports() > kModelPort && ctx.iport_connected(kModelPort))) {
    ensure_loaded_();
  }
  co_return;
}

Job
FlashVsrSrcEncoderStage::process(RuntimeContext& ctx)
{
  if (ctx.num_iports() <= kSourcePort || !ctx.iport_connected(kSourcePort)) {
    session()->error(fmt(
        "FlashVsrSrcEncoderStage('{}'): nothing is wired to the source "
        "iport; inert", this->id()));
    ctx.signal_done();
    co_return;
  }
  // Latch the shared model once, before anything is read: the load needs
  // it and the first clip is already on its way.
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

  auto cb = co_await ctx.read(kSourcePort);
  if (!cb) { ctx.signal_done(); co_return; }
  if (_proj == nullptr && !ensure_loaded_()) {
    // Inert, but the clip is consumed: returning without reading makes
    // the runtime re-invoke immediately and the stage spins a core.
    co_return;
  }
  const auto* tb = dynamic_cast<const TensorBeatPayload*>(cb.get());
  if (tb == nullptr || tb->dtype != TensorBeat::DType::U8 ||
      tb->shape.size() != 4 || tb->shape[1] != 3) {
    session()->warn(fmt(
        "FlashVsrSrcEncoderStage('{}'): expected a planar U8 RGB "
        "[frames, 3, H, W] clip, got {}; dropping beat", this->id(),
        cb->describe()));
    co_return;
  }

  const int frames = (int)tb->shape[0];
  const int H = (int)tb->shape[2], W = (int)tb->shape[3];
  const int dim = _proj->config().out_dim;
  const int sh = _proj->config().shuffle_h, sw = _proj->config().shuffle_w;
  if (H % sh != 0 || W % sw != 0) {
    session()->warn(fmt(
        "FlashVsrSrcEncoderStage('{}'): clip is {}x{}; the projection "
        "unshuffles by {}x{}, so both axes must divide by it",
        this->id(), W, H, sw, sh));
    co_return;
  }
  const std::size_t tok = (std::size_t)(H / sh) * (W / sw);
  const std::size_t frame_px = (std::size_t)3 * H * W;
  const auto bytes = tb->materialize_contiguous();
  if (bytes.size() < frame_px * (std::size_t)frames) {
    session()->warn(fmt(
        "FlashVsrSrcEncoderStage('{}'): clip is short of its own shape",
        this->id()));
    co_return;
  }

  // THE CALL PATTERN IS THE REFERENCE'S, not a simplification. The
  // projection is causal: it carries two frames of each convolution
  // across calls and its OPENING call seeds those carries and produces
  // nothing. Seven slices open the run, the first a single frame; then
  // two per denoise chunk. One call over the whole clip returns no rows.
  _proj->reset();
  std::vector<metal_compute::SharedBuffer> rows;
  std::string err;
  const int chunks = (frames - 1) / 8 - 2;
  auto drive = [&](int lo, int hi) {
    if (hi <= lo || hi > frames) { return true; }
    int rf = 0;
    return _proj->stream_forward(bytes.data() + (std::size_t)lo * frame_px,
                                 hi - lo, H, W, &rows, &rf, &err);
  };
  bool ok = true;
  for (int j = 0; j < 7 && ok; ++j) {
    ok = drive(std::max(0, j * 4 - 3), (j + 1) * 4 - 3);
  }
  for (int c = 1; c < chunks && ok; ++c) {
    for (int j = 0; j < 2 && ok; ++j) {
      ok = drive(c * 8 + 17 + j * 4, c * 8 + 21 + j * 4);
    }
  }
  if (!ok) {
    session()->warn(fmt(
        "FlashVsrSrcEncoderStage('{}'): source projection failed: {}",
        this->id(), err));
    co_return;
  }
  if (std::getenv("VPIPE_FVSR_ENC_LOG") != nullptr) {
    const auto ms = metal_compute::shared_buffer_memory_stats();
    session()->info(fmt(
        "FlashVsrSrcEncoderStage('{}'): after a {}-frame clip: live {} MB, "
        "peak {} MB, {} row frames", this->id(), frames, ms.live_bytes >> 20,
        ms.peak_bytes >> 20, rows.size()));
  }
  if (rows.empty()) {
    session()->warn(fmt(
        "FlashVsrSrcEncoderStage('{}'): a {}-frame clip produced no rows -- "
        "twenty frames go to a warmup that is never emitted, so 25 is the "
        "shortest clip this family restores anything from",
        this->id(), frames));
    co_return;
  }

  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::Bf16;
  out->shape = {(std::int64_t)(rows.size() * tok), (std::int64_t)dim};
  out->resize_contiguous(rows.size() * tok * (std::size_t)dim);
  auto* dst = (std::uint8_t*)out->as_bf16();
  for (std::size_t i = 0; i < rows.size(); ++i) {
    std::memcpy(dst + i * tok * (std::size_t)dim * 2, rows[i].contents(),
                tok * (std::size_t)dim * 2);
  }
  // The geometry the rows belong to. A row count alone is ambiguous --
  // the same number comes out of several (size, frames) pairs -- and the
  // denoiser has to slice these per chunk.
  {
    // A fresh payload's sideband is NULL, and as_object() on it throws --
    // after the projection has already done all its work.
    out->sideband = FlexData::make_object();
    auto o = out->sideband.as_object();
    o.insert_or_assign("source_width", FlexData::make_int(W));
    o.insert_or_assign("source_height", FlexData::make_int(H));
    o.insert_or_assign("source_frames", FlexData::make_int(frames));
    o.insert_or_assign("row_frames", FlexData::make_int((int)rows.size()));
  }
  session()->log_debug(fmt(
      "FlashVsrSrcEncoderStage('{}'): {}x{} x{} frames -> {} row frames x {} "
      "tokens x {}", this->id(), W, H, frames, rows.size(), tok, dim));
  co_await ctx.write(0, std::move(out));
  ++_emitted;
}

#else   // !VPIPE_BUILD_APPLE_SILICON

std::vector<ResourceClaim>
FlashVsrSrcEncoderStage::declare_resources() const
{
  return {};
}

StageMemory
FlashVsrSrcEncoderStage::declare_memory() const
{
  return {};
}

void
FlashVsrSrcEncoderStage::reset_run_state()
{
}

Job
FlashVsrSrcEncoderStage::initialize(RuntimeContext&)
{
  session()->error(fmt(
      "FlashVsrSrcEncoderStage('{}'): the source projection runs on the "
      "metal-compute backend; this build has none", this->id()));
  co_return;
}

Job
FlashVsrSrcEncoderStage::process(RuntimeContext& ctx)
{
  ctx.signal_done();
  co_return;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

VPIPE_REGISTER_STAGE(FlashVsrSrcEncoderStage)
VPIPE_REGISTER_SPEC(FlashVsrSrcEncoderStage, kSpec)

}  // namespace vpipe
