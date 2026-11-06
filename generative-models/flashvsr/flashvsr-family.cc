#include "generative-models/flashvsr/flashvsr-family.h"
#include "generative-models/flashvsr/flashvsr-tiny-decoder.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/beat-keys.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/flashvsr/flashvsr-lq-proj.h"
#include "generative-models/flashvsr/metal-flashvsr-transformer.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/wan/metal-wan-vae.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-memory.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
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
constexpr const char* kTinyDecoder      = "TCDecoder.ckpt";
// FlashVSR+'s, one pair per variant folder.
constexpr const char* kPlusDenoiser = "dit.safetensors";
constexpr const char* kPlusSource   = "lq_proj_in.safetensors";

// The variant a FlashVSR+ folder holds: its own name when that is one of
// the two, else the repository config's `variants` map read backwards
// (name -> {"path": folder}). Empty when neither says.
std::string
plus_variant_(const fs::path& dir)
{
  std::string base = dir.filename().string();
  std::transform(base.begin(), base.end(), base.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });
  if (base == "turbo" || base == "base") { return base; }
  std::ifstream in(dir.parent_path() / "config.json");
  if (!in) { return {}; }
  try {
    FlexData cfg = FlexData::from_json(in);
    if (!cfg.is_object() || !cfg.as_object().contains("variants")) {
      return {};
    }
    FlexData vs = cfg.as_object().at("variants");
    if (!vs.is_object()) { return {}; }
    for (const auto& [name, v] : vs.as_object()) {
      FlexData one = v;
      if (one.is_object() && one.as_object().contains("path") &&
          std::string(one.as_object().at("path").as_string("")) ==
              dir.filename().string()) {
        std::string n(name);
        std::transform(n.begin(), n.end(), n.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });
        return n == "turbo" || n == "base" ? n : std::string{};
      }
    }
  } catch (...) {
  }
  return {};
}

// A FlashVSR+ variant folder `dir`, or false.
bool
resolve_plus_variant_(const fs::path& dir, FlashVsrLayout* out)
{
  std::error_code ec;
  const fs::path dit = dir / kPlusDenoiser;
  const fs::path src = dir / kPlusSource;
  if (!fs::is_regular_file(dit, ec) || !fs::is_regular_file(src, ec)) {
    return false;
  }
  // The shared files sit beside the variant or one level up, where the
  // repository keeps them.
  auto shared = [&](const char* name) -> std::string {
    for (const fs::path& d : {dir, dir.parent_path()}) {
      if (fs::is_regular_file(d / name, ec)) { return (d / name).string(); }
    }
    return {};
  };
  const std::string ctx = shared(kPublishedContext);
  if (ctx.empty()) { return false; }
  FlashVsrLayout l;
  l.denoiser = dit.string();
  l.source = src.string();
  l.source_prefix = "";
  l.context = ctx;
  l.context_name = "posi_prompt";
  l.vae = shared(kPublishedVae);
  l.tiny_decoder = shared(kTinyDecoder);
  l.published = true;
  l.plus = true;
  l.variant = plus_variant_(dir);
  if (out != nullptr) { *out = std::move(l); }
  return true;
}

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
params_from_config_(const FlexData* fd,
                    const MetalFlashVsrTransformer::Schedule& sch)
{
  // The release's own defaults: v1.1 and FlashVSR+ spell sparse_ratio
  // differently (scaled by the frame area, or not).
  MetalFlashVsrTransformer::Params p =
      MetalFlashVsrTransformer::Params::defaults(sch);
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

// The layout and the denoiser's geometry, with the release's schedule:
// the one read every memory figure here starts from.
bool
config_for_root_(const std::string& root, FlashVsrLayout* layout,
                 MetalFlashVsrTransformer::Config* cfg,
                 std::string* err = nullptr)
{
  if (!resolve_flashvsr_layout(root, layout) ||
      !MetalFlashVsrTransformer::config_from_checkpoint(layout->denoiser, cfg,
                                                        err)) {
    return false;
  }
  cfg->schedule = flashvsr_schedule(*layout);
  return true;
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
    // A STREAM's rows say which of its chunks they hold; a clip's are the
    // whole of one, sized by the stage's frame count.
    const FlexData sb = req.cond_sideband != nullptr ? *req.cond_sideband
                                                     : FlexData{};
    int chunk0 = 0, chunks = 0;
    bool stream = false, final_beat = true;
    if (sb.is_object()) {
      auto o = sb.as_object();
      if (o.contains(flashvsr_stream::kChunk0)) {
        stream = true;
        chunk0 = (int)o.at(flashvsr_stream::kChunk0).as_int(-1);
        chunks = o.contains(flashvsr_stream::kChunks)
                     ? (int)o.at(flashvsr_stream::kChunks).as_int(0)
                     : 0;
        final_beat = o.contains(sideband::kStreamFinal) &&
                     o.at(sideband::kStreamFinal).as_bool(false);
      }
    }
    if (stream && (chunk0 < 0 || chunks <= 0)) {
      return refuse(fmt("stream rows name chunks {}+{}, which is no chunk "
                        "at all", chunk0, chunks)());
    }
    const MetalFlashVsrTransformer::Schedule& sch =
        _dit->config().schedule;
    const int frames =
        MetalFlashVsrTransformer::align_frames(sch, req.frames);
    const int want_t =
        stream ? (chunk0 == 0 ? sch.latents(chunks)
                              : sch.chunk_latents * chunks)
               : MetalFlashVsrTransformer::latent_frames(sch, frames);
    if (want_t <= 0) {
      return refuse(fmt("a {}-frame clip is all warmup; {} frames is the "
                        "shortest that restores anything", req.frames,
                        sch.frames_through(0))());
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
    r.params = params_from_config_(req.model_config, sch);
    if (stream) {
      r.chunk0 = chunk0;
      r.chunks = chunks;
      r.keep_window = !final_beat;
    }
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
    // What the DECODE needs to know, forwarded as the encoder said it:
    // whether this latent continues the last (so its carries do), whether
    // the stream ends here, how many of its frames are real, how runs of
    // repeated frames merge back, the colour fix's references, and how it
    // overlaps its neighbours.
    if (sb.is_object()) {
      FlexData fw = FlexData::make_object();
      auto in = sb.as_object();
      auto o = fw.as_object();
      for (std::string_view k :
           {sideband::kStreamContinues, sideband::kStreamFinal,
            sideband::kValidFrames, sideband::kColorRef,
            sideband::kBlendHead, sideband::kBlendTail,
            sideband::kMergeWeights, sideband::kMergeLead,
            sideband::kColorRefFrames, sideband::kColorRefLevels,
            sideband::kDecoder, sideband::kDecoderCondFrames}) {
        if (in.contains(k)) { o.insert_or_assign(k, in.at(k)); }
      }
      out->sideband = std::move(fw);
    }
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
    if (fs::is_regular_file(r / kTinyDecoder, ec)) {
      l.tiny_decoder = (r / kTinyDecoder).string();
    }
    if (out != nullptr) { *out = std::move(l); }
    return true;
  }

  // FlashVSR+: a variant folder, or the repository holding them -- Turbo
  // when it is there, as the reference's own CLI defaults to it.
  if (resolve_plus_variant_(r, out)) { return true; }
  for (const char* v : {"turbo", "base"}) {
    if (fs::is_regular_file(r / "config.json", ec) &&
        resolve_plus_variant_(r / v, out)) {
      if (out != nullptr) { out->variant = v; }
      return true;
    }
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
  if (fs::is_regular_file(r / kTinyDecoder, ec)) {
    l.tiny_decoder = (r / kTinyDecoder).string();
  }
  l.published = true;
  if (out != nullptr) { *out = std::move(l); }
  return true;
}

MetalFlashVsrTransformer::Schedule
flashvsr_schedule(const FlashVsrLayout& layout)
{
  using S = MetalFlashVsrTransformer::Schedule;
  if (!layout.plus) { return S::v1_1(); }
  return S::plus(layout.variant == "turbo");
}

MetalFlashVsrTransformer::Schedule
flashvsr_schedule(const std::string& root)
{
  FlashVsrLayout l;
  if (!resolve_flashvsr_layout(root, &l)) {
    return MetalFlashVsrTransformer::Schedule::v1_1();
  }
  return flashvsr_schedule(l);
}

namespace {

// The tiny decoder's weights and scratch at this geometry, when the
// layout has one and asked for: a stream's beats are whole chunks, and
// the scratch past one commit's worth does not grow with them.
bool
tiny_figures_(const FlashVsrLayout& layout, int width, int height,
              std::size_t* weights, std::size_t* scratch)
{
  return !layout.tiny_decoder.empty() &&
         FlashVsrTinyDecoder::plan(layout.tiny_decoder, 4, height / 8,
                                   width / 8, weights, scratch);
}

}  // namespace

std::size_t
flashvsr_stream_bytes(const std::string& root, int width, int height,
                      double kv_ratio, bool tiny)
{
  if (width <= 0 || height <= 0) { return 0; }
  FlashVsrLayout layout;
  MetalFlashVsrTransformer::Config cfg;
  if (!config_for_root_(root, &layout, &cfg)) { return 0; }
  MetalFlashVsrTransformer::Params prm =
      MetalFlashVsrTransformer::Params::defaults(cfg.schedule);
  prm.kv_ratio = kv_ratio;
  const std::size_t windows =
      (std::size_t)MetalFlashVsrTransformer::stream_window_bytes(
          cfg, height, width, prm);
  std::size_t tw = 0, ts = 0;
  if (tiny && tiny_figures_(layout, width, height, &tw, &ts)) {
    return windows + ts;
  }
  // The VAE is Wan 2.1's; read its config where the checkpoint has one.
  MetalWanVae::Config vcfg;
  if (!layout.vae.empty() &&
      !MetalWanVae::config_from_json(layout.vae, vcfg, nullptr)) {
    MetalWanVae::config_for_native_checkpoint(layout.vae, vcfg);
  }
  return windows +
         MetalWanVae::decode_carry_bytes(vcfg, height / 8, width / 8);
}

bool
flashvsr_stream_fits(const std::string& root, int width, int height,
                     std::size_t* need, bool tiny)
{
  if (need != nullptr) { *need = 0; }
  const std::size_t carries =
      flashvsr_stream_bytes(root, width, height, 3.0, tiny);
  if (carries == 0) { return false; }
  FlashVsrLayout layout;
  MetalFlashVsrTransformer::Config cfg;
  if (!config_for_root_(root, &layout, &cfg)) { return false; }
  // The rest of the decode beside its carries: the tiny decoder's
  // weights (its scratch is all carried), or a whole-plane VAE decode's
  // working set past its carries.
  std::size_t tw = 0, ts = 0;
  const bool small = tiny && tiny_figures_(layout, width, height, &tw, &ts);
  MetalWanVae::Config vcfg;
  if (!layout.vae.empty() &&
      !MetalWanVae::config_from_json(layout.vae, vcfg, nullptr)) {
    MetalWanVae::config_for_native_checkpoint(layout.vae, vcfg);
  }
  const int h8 = height / 8, w8 = width / 8;
  const std::size_t peak =
      small ? tw : MetalWanVae::decode_peak_bytes(vcfg, h8, w8);
  const std::size_t vae_carries =
      small ? 0 : MetalWanVae::decode_carry_bytes(vcfg, h8, w8);
  // Beside the carries, what a stream cannot do without: the denoiser at
  // its STREAMING floor (a tight box streams its blocks, and a stream
  // asks nothing of their residency), the projection and its working set
  // for a group, and the rest of a whole-plane decode -- a carried decode
  // cannot tile -- the carries being counted already.
  const std::size_t total =
      carries +
      (std::size_t)MetalFlashVsrTransformer::streaming_floor_bytes(cfg) +
      FlashVsrLqProj::weight_bytes(layout.source, layout.source_prefix) +
      FlashVsrLqProj::working_bytes(layout.source, layout.source_prefix,
                                    height, width,
                                    cfg.schedule.latents(1)) +
      (peak > vae_carries ? peak - vae_carries : 0);
  if (need != nullptr) { *need = total; }
  const std::size_t ram = model_memory::phys_ram();
  if (ram == 0) { return true; }
  return total + model_memory::kStreamHeadroom <= ram;
}

bool
FlashVsrVideoFamily::claims(const std::string& root,
                            const std::string& model_type) const
{
  if (model_type == "flashvsr") { return true; }
  return resolve_flashvsr_layout(root, nullptr);
}

int
FlashVsrVideoFamily::align_frames(const std::string& root, int frames) const
{
  return MetalFlashVsrTransformer::align_frames(flashvsr_schedule(root),
                                                frames);
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
  if (config_for_root_(root, &layout, &cfg)) {
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
FlashVsrVideoFamily::latent_bytes(const std::string& root, int width,
                                  int height, int frames) const
{
  const auto sch = flashvsr_schedule(root);
  const int t = MetalFlashVsrTransformer::latent_frames(
      sch, MetalFlashVsrTransformer::align_frames(sch, frames));
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
  if (!config_for_root_(root, &layout, &cfg)) { return 0; }
  // ...AND THE SOURCE PROJECTION'S, which a pipelined graph runs for the
  // NEXT clip while this one denoises: the conditioning stage sees one
  // clip at a time and cannot know the geometry before its first beat,
  // and in steady state its working set is live in this phase, not only
  // in its own. 901 MB at 1920x1152 x 25 that the plan did not have;
  // MEASURED 748 MB above the weights on a first clip, before any peer
  // had allocated.
  // ...AND A STREAM'S WINDOWS, held from one clip's denoise to the next
  // whenever the source encoder streams -- which, left to decide, it does
  // exactly when flashvsr_stream_fits() says so, the same rule asked the
  // same way.
  const MetalFlashVsrTransformer::Params prm =
      params_from_config_(model_config, cfg.schedule);
  const std::size_t windows =
      flashvsr_stream_fits(root, width, height)
          ? (std::size_t)MetalFlashVsrTransformer::stream_window_bytes(
                cfg, height, width, prm)
          : 0;
  return (std::size_t)MetalFlashVsrTransformer::denoise_scratch_bytes(
             cfg, height, width, frames, prm) +
         windows +
         FlashVsrLqProj::working_bytes(
             layout.source, layout.source_prefix, height, width,
             MetalFlashVsrTransformer::latent_frames(
                 cfg.schedule, MetalFlashVsrTransformer::align_frames(
                                   cfg.schedule, frames)));
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
    return fail(fmt("'{}' holds neither a converted FlashVSR checkpoint, "
                    "the published v1.1 files ({}, {}, {}), nor a "
                    "FlashVSR+ variant ({} and {} beside {})", root,
                    kPublishedDenoiser, kPublishedSource,
                    kPublishedContext, kPlusDenoiser, kPlusSource,
                    kPublishedContext)());
  }
  // TURBO AND BASE ARE THE SAME TENSORS; only the folder says which, and
  // running one's weights with the other's attention layout is a quietly
  // worse picture. So an unnamed one is refused, not guessed.
  if (layout.plus && layout.variant.empty()) {
    return fail(fmt("'{}' is a FlashVSR+ variant, but neither its folder "
                    "name nor the repository's config.json says whether "
                    "it is turbo or base -- keep it in a folder of that "
                    "name, as the repository publishes it", root)());
  }
  MetalFlashVsrTransformer::Config cfg;
  std::string err;
  if (!MetalFlashVsrTransformer::config_from_checkpoint(layout.denoiser, &cfg,
                                                        &err)) {
    return fail("cannot read the checkpoint's geometry: " + err);
  }
  cfg.schedule = flashvsr_schedule(layout);
  // What the graph asked for. Both are decided at LOAD -- int8 sizes a
  // scratch and Sage builds a kernel variant -- so they come off the
  // create args, which the stage fills from the same config as the
  // per-generation bag.
  cfg.i8_gemm = accel::flag(args.accel, accel::kI8Gemm, cfg.i8_gemm);
  cfg.autotune = accel::flag(args.accel, accel::kAutotune, cfg.autotune);
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
        "flashvsr: {} -- {} blocks at {} wide, {}x{} heads; one step at a "
        "fixed timestep over a constant conditioning; chunks of {} latent "
        "frames{}",
        layout.plus ? "FlashVSR+ " + layout.variant : std::string("v1.1"),
        cfg.n_layers, cfg.hidden, cfg.n_heads, cfg.head_dim,
        cfg.schedule.chunk_latents,
        cfg.schedule.sliding_layers.empty()
            ? std::string()
            : fmt(", {} hybrid layers (3x3 band, one window kept)",
                  cfg.schedule.sliding_layers.size())()));
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
