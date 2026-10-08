#include "generative-models/flashvsr/flashvsr-family.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/vpipe-format.h"
#include "generative-models/flashvsr/flashvsr-lq-proj.h"
#include "generative-models/flashvsr/metal-flashvsr-transformer.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-memory.h"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vpipe {
namespace genai {

namespace fs = std::filesystem;

namespace {

// The size grid and the frame rule, restated here because the FAMILY is
// asked about them before anything is loaded.
constexpr int kGrid = MetalFlashVsrTransformer::kSizeGrid;

// The published file names. The source projection is what a layout is
// recognised BY, and not the denoiser: the denoiser alone is Wan
// 2.1-T2V-1.3B, so claiming on it would take a plain Wan checkpoint and
// drive it with a projection it does not have -- the worst failure this
// family has, because it loads at full cost and computes nonsense.
constexpr const char* kPublishedDenoiser =
    "diffusion_pytorch_model_streaming_dmd.safetensors";
constexpr const char* kPublishedSource  = "LQ_proj_in.ckpt";
constexpr const char* kPublishedContext = "posi_prompt.pth";
constexpr const char* kPublishedVae     = "Wan2.1_VAE.pth";

// The sets the DENOISER reads: its own, plus the context's when that is a
// separate file. The source projection is the source encoder stage's
// declaration, not this family's.
std::vector<std::string>
denoiser_sets_(const std::string& root)
{
  FlashVsrLayout l;
  if (!resolve_flashvsr_layout(root, &l)) { return {root}; }
  if (l.context == l.denoiser) { return {l.denoiser}; }
  return {l.denoiser, l.context};
}

// The family's own knobs, off the model-config beat. Absent keys keep the
// reference's documented defaults, and so does a value the routing
// cannot use -- the source stage refuses those at config time, so one
// arriving here came from somewhere that did not check. ONE parser, for
// the generation and for the memory plan, so the window a plan sized is
// the window the forward allocates.
MetalFlashVsrTransformer::Params
params_from_config_(const FlexData* fd)
{
  MetalFlashVsrTransformer::Params p;
  if (fd == nullptr || !fd->is_object()) { return p; }
  auto o = fd->as_object();
  if (o.contains("sparse_ratio")) {
    const double v = o.at("sparse_ratio").as_real(p.sparse_ratio);
    if (v > 0.0) { p.sparse_ratio = v; }
  }
  if (o.contains("kv_ratio")) {
    const double v = o.at("kv_ratio").as_real(p.kv_ratio);
    if (v >= 0.0) { p.kv_ratio = v; }
  }
  if (o.contains("local_range")) {
    const int v = (int)o.at("local_range").as_int(p.local_range);
    if (v >= 1) { p.local_range = v; }
  }
  return p;
}

class FlashVsrGenerator final : public VideoGenerator {
 public:
  FlashVsrGenerator(std::unique_ptr<MetalFlashVsrTransformer> dit,
                    const SessionContextIntf* session)
      : _dit(std::move(dit)), _session(session),
        _hidden(_dit != nullptr ? _dit->config().hidden : 0) {}

  int latent_channels() const override { return 16; }
  int spatial_compression() const override { return 8; }

  bool
  generate(const VideoGenRequest& req, VideoGenResult* out) override
  {
    auto refuse = [&](const std::string& why) {
      if (_session != nullptr) {
        _session->warn(fmt("flashvsr: {}", why));
      }
      return false;
    };
    if (_dit == nullptr || out == nullptr) { return refuse("not loaded"); }

    // THE SOURCE IS THE CONDITIONING. It arrives on the conditioning
    // port as rows, because the conditioner ran this family's source
    // projection over the clip -- so an unwired conditioner is exactly
    // the refusal a missing prompt would be for any other family, and
    // not a fallback. Generating anyway would produce a confident 4x
    // upscale of noise, which looks like a working model.
    if (req.cond == nullptr || req.cond_rows <= 0) {
      return refuse("conditions on its source clip and nothing else; wire a "
                    "diffusion-conditioner on the same model directory, with "
                    "the clip on its source_video port");
    }
    if (req.cond_dim != _hidden) {
      return refuse(fmt("conditioning is {} wide; this checkpoint's rows are "
                        "{} -- the conditioner resolved a different model",
                        req.cond_dim, _hidden)());
    }
    // The geometry the rows describe. Width and height come from the
    // stage, which took them from the clip the conditioner resampled;
    // the row count then has to agree, and saying so here beats
    // discovering it as a short read forty layers in.
    const int H = req.height, W = req.width;
    if (H <= 0 || W <= 0) {
      return refuse("no output geometry; generate-video's width and height "
                    "describe the clip this family restores");
    }
    if (H % kGrid != 0 || W % kGrid != 0) {
      return refuse(fmt("output is {}x{}; both axes must be multiples of {} "
                        "(the window partition needs the token grid to "
                        "divide by 8, and the grid is the frame over 16)",
                        W, H, kGrid)());
    }
    const int frames = MetalFlashVsrTransformer::align_frames(req.frames);
    const int want_t = MetalFlashVsrTransformer::latent_frames(frames);
    if (want_t <= 0) {
      return refuse(fmt("a {}-frame clip is all warmup; 25 frames is the "
                        "shortest that restores anything", req.frames)());
    }
    const int tok = (H / 16) * (W / 16);
    if (tok <= 0 || req.cond_rows % tok != 0) {
      return refuse(fmt("conditioning has {} rows, which is not a whole "
                        "number of {}-token frames at {}x{}",
                        req.cond_rows, tok, W, H)());
    }
    const int row_frames = req.cond_rows / tok;
    if (row_frames < want_t) {
      return refuse(fmt("conditioning covers {} frames; a {}-frame clip needs "
                        "{}", row_frames, frames, want_t)());
    }

    MetalFlashVsrTransformer::Request r;
    r.rows = req.cond;
    r.row_frames = row_frames;
    r.frames = frames;
    r.height = H;
    r.width = W;
    r.seed = req.seed;
    r.params = params_from_config_(req.model_config);
    // One step at a fixed timestep, so the configured step count has
    // nothing to select. The bar follows the unit of work instead: a
    // block's pass over the whole clip (block-outer, 30 of them, chunks
    // within), or a chunk (chunk-outer, blocks within).
    r.progress = req.progress;
    r.block_progress = req.block_progress;

    std::string err;
    if (!_dit->generate(r, &out->video, &out->video_shape, &err)) {
      return refuse(err.empty() ? std::string("denoise failed") : err);
    }
    out->latents_per_second = 0.0;
    return true;
  }

  void release_idle() override {}

  std::uint64_t
  resident_bytes() const override
  {
    return _dit != nullptr ? _dit->resident_bytes() : 0;
  }

 private:
  std::unique_ptr<MetalFlashVsrTransformer> _dit;
  const SessionContextIntf*                 _session = nullptr;
  int                                       _hidden  = 0;
};

}  // namespace

bool
resolve_flashvsr_layout(const std::string& root, FlashVsrLayout* out)
{
  std::error_code ec;
  const fs::path r(root);
  if (!fs::is_directory(r, ec) || ec) { return false; }
  FlashVsrLayout l;

  // CONVERTED first: a directory the converter ran over still holds the
  // published files beside its index, and the index names the same bytes.
  auto w = MetalLlamaWeights::open_model(root);
  if (w.has_value() && w->has("lq_proj.conv1.weight") &&
      w->has("posi_context")) {
    l.denoiser = root;
    l.source = root;
    l.source_prefix = "lq_proj.";
    l.context = root;
    l.context_name = "posi_context";
    if (fs::is_directory(r / "vae", ec)) { l.vae = (r / "vae").string(); }
    if (out != nullptr) { *out = std::move(l); }
    return true;
  }

  // PUBLISHED: recognised by its files, each opened by path.
  const fs::path dit = r / kPublishedDenoiser;
  const fs::path src = r / kPublishedSource;
  const fs::path ctx = r / kPublishedContext;
  if (!fs::is_regular_file(dit, ec) || !fs::is_regular_file(src, ec) ||
      !fs::is_regular_file(ctx, ec)) {
    return false;
  }
  l.denoiser = dit.string();
  l.source = src.string();
  l.context = ctx.string();
  l.context_name = "posi_prompt";
  if (fs::is_regular_file(r / kPublishedVae, ec)) {
    l.vae = (r / kPublishedVae).string();
  }
  l.published = true;
  if (out != nullptr) { *out = std::move(l); }
  return true;
}

bool
FlashVsrVideoFamily::claims(const std::string& root,
                            const std::string& model_type) const
{
  if (model_type == "flashvsr") { return true; }
  return resolve_flashvsr_layout(root, nullptr);
}

int
FlashVsrVideoFamily::align_frames(const std::string&, int frames) const
{
  return MetalFlashVsrTransformer::align_frames(frames);
}

void
FlashVsrVideoFamily::size_grid(const std::string&, int* gh, int* gw) const
{
  if (gh != nullptr) { *gh = kGrid; }
  if (gw != nullptr) { *gw = kGrid; }
}

std::vector<ResourceClaim>
FlashVsrVideoFamily::declare_resources(const std::string& root) const
{
  // The denoiser STREAMS when the box is tight, so its claim carries the
  // floor it streams at -- the claim is what the plan refuses a graph by.
  const auto sets = denoiser_sets_(root);
  const std::vector<StageHolding> h = declare_holdings(root);
  if (h.empty() || h[0].floor >= h[0].preload) {
    return model_memory::weight_claims(sets);
  }
  std::vector<ResourceClaim> out;
  out.push_back(model_memory::weight_claim_streamable(sets[0], h[0].floor));
  for (std::size_t i = 1; i < sets.size(); ++i) {
    for (auto& c : model_memory::weight_claims({sets[i]})) {
      out.push_back(std::move(c));
    }
  }
  return out;
}

std::vector<StageHolding>
FlashVsrVideoFamily::declare_holdings(const std::string& root) const
{
  // THE FLOOR IS WHAT A STREAMING LOAD HOLDS -- the folds, two block
  // slots and the narrowing scratch, ~0.3 GB -- and it is a promise kept:
  // the load streams whenever the stage says the box is tight, which is
  // the only time a floor is what a plan reads.
  // NAMED BY THE DENOISER'S OWN SET, not the root: this is the name the
  // stage releases by, and drop_weights matches a set's key exactly. The
  // published layout opens the denoiser by FILE, so a holding named for
  // the directory released nothing -- MEASURED at 1920x1152 on a 24 GB box,
  // the finished 1.3B DiT's bf16 weights stayed live (3.69 GB of GPU
  // buffers at the decode) and the decode was refused short of room.
  StageHolding h;
  h.source = denoiser_sets_(root).front();
  h.preload = model_memory::weight_footprint(nullptr, denoiser_sets_(root));
  h.floor = h.preload;
  FlashVsrLayout layout;
  MetalFlashVsrTransformer::Config cfg;
  if (resolve_flashvsr_layout(root, &layout) &&
      MetalFlashVsrTransformer::config_from_checkpoint(layout.denoiser,
                                                       &cfg)) {
    std::size_t floor =
        (std::size_t)MetalFlashVsrTransformer::streaming_floor_bytes(cfg);
    // A CONVERTED directory is one set: the projection and the fixed
    // context live in it beside the denoiser, never stream, and are held
    // for the whole run. The source encoder's holding names the same set,
    // and same-named holdings merge by their LARGER floor -- so if this
    // floor left them out, they were lost from every phase at the floor.
    // MEASURED: the plan's floor read 302 MB while the set held 549 MB of
    // projection on top of the streamed denoiser.
    if (layout.source == layout.denoiser) {
      floor += FlashVsrLqProj::weight_bytes(layout.source,
                                            layout.source_prefix);
    }
    if (layout.context == layout.denoiser) {
      floor += FlashVsrLqProj::weight_bytes(layout.context,
                                            layout.context_name);
    }
    h.floor = std::min<std::size_t>(h.preload, floor);
  }
  return {h};
}

std::size_t
FlashVsrVideoFamily::latent_bytes(const std::string&, int width, int height,
                                  int frames) const
{
  const int t = MetalFlashVsrTransformer::latent_frames(frames);
  if (t <= 0 || width <= 0 || height <= 0) { return 0; }
  return (std::size_t)16 * t * (height / 8) * (width / 8) * sizeof(float);
}

std::size_t
FlashVsrVideoFamily::denoise_scratch_bytes(const std::string& root, int width,
                                           int height, int frames,
                                           const FlexData* model_config) const
{
  if (width <= 0 || height <= 0 || frames <= 0) { return 0; }
  FlashVsrLayout layout;
  MetalFlashVsrTransformer::Config cfg;
  // The geometry comes off the checkpoint's own shapes -- one header read,
  // no weights -- so a checkpoint of another width is sized as itself.
  if (!resolve_flashvsr_layout(root, &layout) ||
      !MetalFlashVsrTransformer::config_from_checkpoint(layout.denoiser,
                                                        &cfg)) {
    return 0;
  }
  // ...AND THE SOURCE PROJECTION'S, which a pipelined graph runs for the
  // NEXT clip while this one denoises: the conditioning stage sees one
  // clip at a time and cannot know the geometry before its first beat,
  // and in steady state its working set is live in this phase, not only
  // in its own. 901 MB at 1920x1152 x 25 that the plan did not have;
  // MEASURED 748 MB above the weights on a first clip, before any peer
  // had allocated.
  return (std::size_t)MetalFlashVsrTransformer::denoise_scratch_bytes(
             cfg, height, width, frames, params_from_config_(model_config)) +
         FlashVsrLqProj::working_bytes(layout.source, layout.source_prefix,
                                       height, width, frames);
}

std::unique_ptr<VideoGenerator>
FlashVsrVideoFamily::load(const VideoModelCreateArgs& args)
{
  auto fail = [&](const std::string& why) -> std::unique_ptr<VideoGenerator> {
    if (args.session != nullptr) {
      args.session->error(fmt("flashvsr: {}", why));
    }
    return nullptr;
  };
  if (args.metal == nullptr) { return fail("no metal-compute backend"); }
  const std::string root = args.dit_dir.empty() ? args.root : args.dit_dir;

  FlashVsrLayout layout;
  if (!resolve_flashvsr_layout(root, &layout)) {
    return fail(fmt("'{}' holds neither a converted FlashVSR checkpoint nor "
                    "the published files ({}, {}, {})", root,
                    kPublishedDenoiser, kPublishedSource,
                    kPublishedContext)());
  }
  MetalFlashVsrTransformer::Config cfg;
  std::string err;
  if (!MetalFlashVsrTransformer::config_from_checkpoint(layout.denoiser, &cfg,
                                                        &err)) {
    return fail("cannot read the checkpoint's geometry: " + err);
  }
  // What the graph asked for. Both are decided at LOAD -- int8 sizes a
  // scratch and Sage builds a kernel variant -- so they come off the
  // create args, which the stage fills from the same config as the
  // per-generation bag.
  cfg.i8_gemm = accel::flag(args.accel, accel::kI8Gemm, cfg.i8_gemm);
  cfg.sage = sage::config_from_flex(args.accel);
  // SOL IS NOT A TIER THIS FAMILY TAKES, and a graph that asked for it
  // is told so rather than left to read a wall clock. The self-attention
  // here is already block-sparse under the checkpoint's OWN routing --
  // trained into the weights, not an approximation layered on after --
  // so a second router would be deciding over that decision.
  for (std::string_view t : accel::tiers_on(args.accel)) {
    if (t == accel::kSolAttn && args.session != nullptr) {
      args.session->log_normal(fmt(
          "flashvsr: sol_attn is not implemented here -- the denoiser's "
          "self-attention is already routed by the checkpoint's own "
          "locality-constrained sparsity"));
    }
  }
  // STREAM WHEN THE BOX IS TIGHT, which is the stage's verdict to give:
  // every video model in the process answers the memory question the
  // same way. VPIPE_FVSR_STREAM=1 / 0 overrules it for an A/B.
  cfg.stream_blocks = args.prefer_streaming;
  if (const char* e = std::getenv("VPIPE_FVSR_STREAM")) {
    if (*e != '\0') { cfg.stream_blocks = std::string(e) != "0"; }
  }
  auto ws = open_weight_set(layout.denoiser, args.session);
  if (ws == nullptr) { return fail("cannot open " + layout.denoiser); }
  auto ctx_ws = layout.context == layout.denoiser
                    ? ws : open_weight_set(layout.context, args.session);
  if (ctx_ws == nullptr) { return fail("cannot open " + layout.context); }
  auto dit = MetalFlashVsrTransformer::load(ws, ctx_ws, layout.context_name,
                                            args.metal, cfg, &err);
  if (dit == nullptr) { return fail(err.empty() ? "load failed" : err); }
  if (args.session != nullptr) {
    args.session->log_normal(fmt(
        "flashvsr: {} blocks at {} wide, {}x{} heads; one step at a fixed "
        "timestep over a constant conditioning", cfg.n_layers, cfg.hidden,
        cfg.n_heads, cfg.head_dim));
    // What the graph GOT, which is not always what it asked for: Sage
    // without the NAX entry, or either tier on a box with no matrix
    // cores, runs the plain kernels and says so here.
    args.session->log_normal(fmt("flashvsr: {}", dit->accel_summary()));
    if (dit->streams_blocks()) {
      args.session->log_normal(fmt(
          "flashvsr: streaming its {} blocks from the checkpoint, one read "
          "per block per clip; {} MB held", cfg.n_layers,
          dit->resident_bytes() >> 20));
    }
  }
  return std::make_unique<FlashVsrGenerator>(std::move(dit), args.session);
}

void
register_flashvsr_video_family()
{
  // First-wins on the tag, so a second call is a no-op rather than a
  // shadowing surprise.
  VideoModelRegistry::get().add(std::make_unique<FlashVsrVideoFamily>());
}

namespace {
// THE FIRST IN-TREE REGISTERED VIDEO FAMILY. Everything before this
// registered from a plugin, and the built-in wan / minimax-h3 branches
// are not registered at all. Same idiom as VPIPE_REGISTER_RESOURCE_PLANNER:
// libvpipe is a shared library, so this runs at load and the registry it
// reaches is the one the stage consults.
const int _flashvsr_family_reg = [] {
  register_flashvsr_video_family();
  return 0;
}();
}  // namespace

}  // namespace genai
}  // namespace vpipe
