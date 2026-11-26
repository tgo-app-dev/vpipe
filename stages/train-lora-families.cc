// The model families train-lora trains (train::lora_families()): for
// each, how a checkpoint is recognised, what it holds and what a step
// holds -- the plan the stage's stream and keep-all decisions and its
// claims are made from, read off the checkpoint before anything loads --
// and the load.
//
// Here, in the stage layer, because the plan is the memory planner's
// arithmetic (model_memory::dir_weights_bytes, streaming floors) over the
// model classes' own figures.

#include "generative-models/train/lora-trainer.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-image.h"
#include "generative-models/minimax-h3/minimax-h3-lora-trainer.h"
#include "generative-models/minimax-h3/p-h3-image.h"
#include "generative-models/qwen-image/qwen-image21-lora-trainer.h"
#include "generative-models/train/optimizer.h"
#include "generative-models/weight-set.h"
#include "stages/model-memory.h"

#include <cstdlib>
#include <filesystem>
#include <string>

namespace vpipe {
namespace genai {
namespace train {

namespace {

namespace fs = std::filesystem;

// The adapter's state a parameter: f32 master and gradient, the bf16
// shadow, and the optimizer's own.
std::size_t
state_bytes_(std::size_t params, const RunShape& r)
{
  return params * (10 + r.opt_bytes_per_param);
}

// ---- Qwen-Image-2.1 ----------------------------------------------------

class QwenImage21Family final : public LoraFamily {
 public:
  const char* name() const noexcept override { return "qwen-image-21"; }

  static std::string
  dit_(const std::string& root)
  {
    return (fs::path(root) / "transformer").string();
  }

  bool
  claims(const std::string& root, const std::string&, const RunShape&,
         std::string*) const override
  {
    return QwenImage21LoraTrainer::claims(dit_(root));
  }

  bool
  claim(const std::string& root, const std::string&, const RunShape&,
        std::string* dit, std::size_t* floor) const override
  {
    *dit = dit_(root);
    *floor = model_memory::streaming_floor_bytes(*dit,
                                                 {"transformer_blocks."});
    return true;
  }

  static bool
  config_(const std::string& dit, const RunShape& r,
          MetalQwenImage21Transformer::Config* cfg, std::string* err)
  {
    if (!MetalQwenImage21Transformer::Config::read_dims(dit, cfg)) {
      *err = "'" + dit + "' has no Qwen-Image-2.1 config";
      return false;
    }
    cfg->ane_ffn = r.ane;
    cfg->ane_qkv = r.ane;
    cfg->i8_gemm = r.i8;
    return true;
  }

  bool
  plan(const std::string& root, const std::string&, const RunShape& r,
       LoadPlan* out, std::string* err) const override
  {
    const std::string dit = dit_(root);
    MetalQwenImage21Transformer::Config cfg;
    if (!config_(dit, r, &cfg, err)) { return false; }
    // A RANK OF A SPLIT RUN: its share of every block's seven projections
    // -- the weights, the step's intermediates and the adapter at the
    // share's widths -- every block recomputed, on the GPU alone.
    const bool rank = r.tp.active();
    if (rank) {
      if (r.tp.n_heads != cfg.n_heads ||
          r.tp.n_ffn != cfg.hidden * cfg.mlp_ratio) {
        *err = fmt("the share is of {} heads and {} rows, the DiT has {} "
                   "and {}", r.tp.n_heads, r.tp.n_ffn, cfg.n_heads,
                   cfg.hidden * cfg.mlp_ratio)();
        return false;
      }
      if (r.ane) {
        *err = "a rank of a split run trains on the GPU alone (no `ane`)";
        return false;
      }
      cfg.ane_ffn = cfg.ane_qkv = false;
      cfg.tp = r.tp;
    }
    out->dit_dir = dit;
    // THE ADAPTER'S STATE, per parameter -- from the dims, before
    // anything loads; a rank's at its share's widths.
    const std::size_t H = (std::size_t)cfg.hidden;
    const std::size_t HI =
        rank ? (std::size_t)r.tp.heads() * (std::size_t)cfg.head_dim : H;
    const std::size_t FF =
        rank ? (std::size_t)r.tp.ffn() : H * (std::size_t)cfg.mlp_ratio;
    const std::size_t per_block =
        r.targets == "attn" ? 4 * (H + HI) : 4 * (H + HI) + 3 * (H + FF);
    out->state_bytes = state_bytes_(per_block * (std::size_t)r.rank *
                                        (std::size_t)cfg.n_layers, r);
    if (rank) {
      std::size_t floor = 0;
      out->held_bytes =
          MetalQwenImage21Transformer::tp_held_bytes(dit, cfg, nullptr,
                                                     &floor);
      out->floor_bytes = floor;
      out->note = fmt("RANK {} of {}: heads [{}, {}) of {}, feed-forward "
                      "rows [{}, {}) of {}", r.tp.rank, r.tp.ranks,
                      r.tp.head0, r.tp.head1, r.tp.n_heads, r.tp.ffn0,
                      r.tp.ffn1, r.tp.n_ffn)();
    } else {
      out->held_bytes = model_memory::dir_weights_bytes(dit);
      out->floor_bytes =
          model_memory::streaming_floor_bytes(dit, {"transformer_blocks."});
    }
    out->arena_recompute = QwenImage21Trainer::arena_bytes(
        cfg, r.max_cells + r.max_text, r.rank, false);
    out->arena_keep = rank ? 0
                           : QwenImage21Trainer::arena_bytes(
                                 cfg, r.max_cells + r.max_text, r.rank, true);
    return true;
  }

  bool
  split_dims(const std::string& root, const std::string&, const RunShape& r,
             int* n_heads, int* n_ffn, std::string* err) const override
  {
    MetalQwenImage21Transformer::Config cfg;
    if (!config_(dit_(root), r, &cfg, err)) { return false; }
    *n_heads = cfg.n_heads;
    *n_ffn = cfg.hidden * cfg.mlp_ratio;
    return true;
  }

  std::unique_ptr<LoraTrainer>
  load(const LoadArgs& a, std::string* err) const override
  {
    QwenImage21LoraTrainer::LoadArgs la;
    la.mc = a.mc;
    la.ws = a.ws;
    la.stream_blocks = a.stream_blocks;
    if (!config_(a.plan->dit_dir, *a.run, &la.cfg, err)) { return nullptr; }
    // A rank's share (RunShape::tp): the load narrows the model to it.
    if (a.run->tp.active()) {
      la.cfg.tp = a.run->tp;
      la.cfg.ane_ffn = la.cfg.ane_qkv = false;
      la.floor_rows = a.run->max_cells + a.run->max_text;
    }
    return QwenImage21LoraTrainer::load(la, err);
  }
};

// ---- MiniMax-H3 --------------------------------------------------------

class MiniMaxH3Family final : public LoraFamily {
 public:
  using M = MetalMiniMaxH3Transformer;

  const char* name() const noexcept override { return "minimax-h3"; }

  static std::string
  partition_(const std::string& model_type)
  {
    return M::partition_of_model_type(model_type);
  }

  using V = MiniMaxH3LoraTrainer::Variant;

  static V
  variant_(const std::string& root)
  {
    return h3i::is_root(root) ? V::kH3i
           : ph3::is_root(root) ? V::kPh3
                                : V::kVideo;
  }

  // The DiT that runs: the root's, or -- for a root whose pipeline names
  // a DiT it does not carry (P-H3-Image's) -- the stage's `dit_dir`.
  static std::string
  dit_of_(const std::string& root, const std::string& model_type,
          const RunShape& r)
  {
    const std::string part = partition_(model_type);
    M::Config probe;
    if (variant_(root) == V::kPh3 && !r.dit_dir.empty() &&
        !M::config_from_json(root, probe, nullptr, part)) {
      return M::resolve_dit_dir(r.dit_dir, part);
    }
    if (!r.dit_dir.empty() && variant_(root) == V::kVideo) {
      return M::resolve_dit_dir(r.dit_dir, part);
    }
    return M::resolve_dit_dir(root, part);
  }

  bool
  claims(const std::string& root, const std::string& model_type,
         const RunShape& r, std::string* err) const override
  {
    const V v = variant_(root);
    M::Config cfg;
    if (M::config_from_json(dit_of_(root, model_type, r), cfg, nullptr,
                            partition_(model_type))) {
      if (v == V::kPh3 && r.base_loras.empty() &&
          !M::config_from_json(root, cfg, nullptr)) {
        *err = "P-H3-Image is FL2VA under PrunaAI's adapter: name the "
               "adapter in base_lora (and FL2VA's DiT in dit_dir)";
        return false;
      }
      return true;
    }
    if (v == V::kPh3) {
      *err = "P-H3-Image's root carries no DiT this build reads: name "
             "MiniMax-H3 FL2VA's in dit_dir and PrunaAI's adapter in "
             "base_lora";
      return false;
    }
    return false;
  }

  bool
  claim(const std::string& root, const std::string& model_type,
        const RunShape& r, std::string* dit, std::size_t* floor) const override
  {
    *dit = dit_of_(root, model_type, r);
    *floor = M::streaming_floor_bytes(*dit);
    return true;
  }

  std::vector<Extra>
  extra_claims(const std::string& root, const std::string& model_type,
               const RunShape& r) const override
  {
    std::vector<Extra> v;
    M::Config cfg;
    const std::string dit = dit_of_(root, model_type, r);
    if (!M::config_from_json(dit, cfg, nullptr, partition_(model_type))) {
      return v;
    }
    // Each frozen adapter as generate-video books one: every factor held,
    // the streamed set its floor.
    for (const std::string& l : r.base_loras) {
      const auto fp = M::lora_footprint(l, cfg.n_layers);
      if (fp.whole > 0) { v.push_back({l, fp.whole, fp.streamed}); }
    }
    if (variant_(root) == V::kH3i) {
      h3i::Manifest man;
      if (h3i::read_manifest(root, &man, nullptr) &&
          !man.heads_file.empty()) {
        const std::size_t b = model_memory::dir_weights_bytes(man.heads_file);
        v.push_back({man.heads_file, b, b});
      }
    }
    return v;
  }

  // The adapter's parameters: rank * (N + K) a projection -- q, k, v
  // [I, H] each and out [H, I]; fc1 [2F, H] and fc2 [H, F] -- over every
  // main block and both token-refiner blocks.
  static std::size_t
  params_(const M::Config& c, const RunShape& r)
  {
    const std::size_t H = (std::size_t)c.hidden, I = (std::size_t)c.inner();
    const std::size_t F = (std::size_t)c.ffn;
    std::size_t per = 4 * (I + H);
    if (r.targets != "attn") { per += (2 * F + H) + (H + F); }
    return per * (std::size_t)r.rank *
           (std::size_t)(c.n_layers + c.n_refiner);
  }

  // What the AdaLN holds pruned at load: every block's [D, 8] f32 and its
  // bias, the curve's basis aside.
  static std::size_t
  pruned_adaln_bytes_(const M::Config& c)
  {
    return (std::size_t)c.n_layers * (std::size_t)c.adaln_out() *
           (8 + 1) * 4;
  }

  bool
  plan(const std::string& root, const std::string& model_type,
       const RunShape& r, LoadPlan* out, std::string* err) const override
  {
    const std::string part = partition_(model_type);
    const std::string dit = dit_of_(root, model_type, r);
    M::Config cfg;
    if (!M::config_from_json(dit, cfg, err, part)) { return false; }
    out->dit_dir = dit;
    const V v = variant_(root);
    if (v == V::kH3i) {
      h3i::Manifest man;
      std::string merr;
      if (!h3i::read_manifest(root, &man, &merr) || man.heads_file.empty()) {
        *err = merr.empty() ? "MiniMax-H3-Image's per-step heads are missing"
                            : merr;
        return false;
      }
      out->aux_dir = man.heads_file;
    }
    // A RANK OF A SPLIT RUN: its share of every block's four projections
    // -- the weights, the step's activations and the adapter at the
    // share's widths -- its AdaLN pruned at load (a rank holds none to
    // run unbaked); its blocks' activations kept where the box holds
    // them, at the share's widths (the stage has the ranks agree).
    const bool rank = r.tp.active();
    if (rank) {
      // The video DiT and MiniMax-H3-Image's (its step heads whole on
      // each rank: they read the residual stream, whole everywhere).
      if (variant_(root) == V::kPh3) {
        *err = "P-H3-Image does not train split across hosts yet";
        return false;
      }
      if (r.tp.n_heads != cfg.n_heads || r.tp.n_ffn != cfg.ffn) {
        *err = fmt("the share is of {} heads and {} rows, the DiT has {} "
                   "and {}", r.tp.n_heads, r.tp.n_ffn, cfg.n_heads,
                   cfg.ffn)();
        return false;
      }
      if (r.ane) {
        *err = "a rank of a split run trains on the GPU alone (no `ane`)";
        return false;
      }
    }
    M::Config tcfg = cfg;
    tcfg.tp = r.tp;
    const M::Config rcfg = M::rank_config(tcfg);
    out->state_bytes = state_bytes_(params_(rcfg, r), r);
    const std::size_t widen = M::fp8_widen_bytes(dit, cfg);
    if (rank) {
      out->arena_recompute = MiniMaxH3LoraTrainer::plan_arena_bytes(
          rcfg, r.max_frames, r.max_grid_h, r.max_grid_w, r.max_text, r.rank,
          widen, r.max_anchors, r.max_cond_rows);
      out->arena_keep = MiniMaxH3LoraTrainer::plan_arena_bytes(
          rcfg, r.max_frames, r.max_grid_h, r.max_grid_w, r.max_text, r.rank,
          widen, r.max_anchors, r.max_cond_rows, true);
      std::size_t base = 0, base_floor = 0;
      for (const std::string& l : r.base_loras) {
        const auto fp = M::lora_footprint(l, cfg.n_layers, nullptr, &r.tp);
        base += fp.whole;
        base_floor += fp.streamed;
      }
      std::size_t floor = 0;
      const std::size_t held =
          M::tp_held_bytes(dit, tcfg, 0, nullptr, &floor);
      const std::size_t adaln = cfg.pruned() ? 0 : pruned_adaln_bytes_(cfg);
      // (MiniMax-H3-Image's per-step heads are booked as an extra
      // folder, whole on every rank, as for the whole model.)
      out->held_bytes = held + base + adaln;
      out->floor_bytes = floor + base_floor + adaln;
      out->note = fmt("RANK {} of {}: heads [{}, {}) of {}, feed-forward rows "
                      "[{}, {}) of {}; AdaLN {}", r.tp.rank, r.tp.ranks,
                      r.tp.head0, r.tp.head1, r.tp.n_heads, r.tp.ffn0,
                      r.tp.ffn1, r.tp.n_ffn,
                      cfg.pruned() ? "already pruned" : "pruned at load")();
      out->opts = FlexData::make_object();
      auto oo = out->opts.as_object();
      oo.insert_or_assign("prune_adaln", FlexData::make_bool(!cfg.pruned()));
      oo.insert_or_assign("dit_partition", FlexData::make_string(part));
      oo.insert_or_assign("partition",
                          FlexData::make_string(M::partition_of(root, part)));
      oo.insert_or_assign("variant", FlexData::make_int((int)variant_(root)));
      return true;
    }
    out->arena_recompute = MiniMaxH3LoraTrainer::plan_arena_bytes(
        cfg, r.max_frames, r.max_grid_h, r.max_grid_w, r.max_text, r.rank,
        widen, r.max_anchors, r.max_cond_rows);
    // recompute: never -- every main block's activations kept.
    out->arena_keep = MiniMaxH3LoraTrainer::plan_arena_bytes(
        cfg, r.max_frames, r.max_grid_h, r.max_grid_w, r.max_text, r.rank,
        widen, r.max_anchors, r.max_cond_rows, true);
    // The frozen adapters beneath, every factor as the binder holds it
    // (and the streamed set a streaming DiT keeps of them).
    std::size_t base = 0, base_floor = 0;
    for (const std::string& l : r.base_loras) {
      const auto fp = M::lora_footprint(l, cfg.n_layers);
      base += fp.whole;
      base_floor += fp.streamed;
    }
    const std::size_t full = model_memory::dir_weights_bytes(dit) + base;
    const std::size_t adaln = cfg.pruned() ? 0 : M::adaln_retired_bytes(dit);
    const std::size_t pruned = full - std::min(full, adaln) +
                               pruned_adaln_bytes_(cfg);
    const std::size_t floor = M::streaming_floor_bytes(dit) + base_floor;
    // THE AdaLN. 13 B of the release's 33 B parameters, which the
    // inference bake retires per schedule and training cannot (a fresh
    // sigma every step). KEPT resident where the whole DiT fits beside a
    // step -- the released modulation exactly -- else PRUNED AT LOAD to
    // the time embedding's 8-dim curve (174 MB; 0.46% from the released
    // forward, closer to fp32), which also takes a streamed block's
    // AdaLN off every step's reads. VPIPE_H3_TRAIN_ADALN=keep|prune is
    // the A/B.
    bool prune = false;
    if (adaln > 0) {
      prune = full + out->arena_recompute + out->state_bytes +
                  r.cache_bytes + r.headroom >
              r.ram;
      if (const char* e = std::getenv("VPIPE_H3_TRAIN_ADALN")) {
        const std::string v(e);
        if (v == "keep") { prune = false; }
        if (v == "prune") { prune = true; }
      }
    }
    out->held_bytes = prune ? pruned : full;
    // A streamed block carries its AdaLN unless it was pruned away.
    out->floor_bytes =
        prune ? floor -
                    std::min(floor, 2 * adaln / std::max(cfg.n_layers, 1)) +
                    pruned_adaln_bytes_(cfg)
              : floor;
    out->note =
        adaln == 0
            ? std::string("the checkpoint's AdaLN is already pruned")
            : prune ? fmt("AdaLN PRUNED AT LOAD ({} MB -> {} MB)",
                          adaln >> 20, pruned_adaln_bytes_(cfg) >> 20)()
                    : fmt("AdaLN kept resident ({} MB)", adaln >> 20)();
    if (r.ane) {
      out->note += "; `ane` is not this family's yet (every GEMM on the "
                   "GPU)";
    }
    out->opts = FlexData::make_object();
    auto oo = out->opts.as_object();
    oo.insert_or_assign("prune_adaln", FlexData::make_bool(prune));
    // The record's partition picks the DiT; the adapter records the one
    // it was trained on, the checkout's own word where no record said.
    oo.insert_or_assign("dit_partition", FlexData::make_string(part));
    oo.insert_or_assign("partition",
                        FlexData::make_string(M::partition_of(root, part)));
    oo.insert_or_assign("variant", FlexData::make_int((int)v));
    if (v != V::kVideo) {
      out->note += v == V::kPh3 ? "; P-H3-Image (one frame, its shift-1 grid)"
                                : "; MiniMax-H3-Image (audio-less, a release "
                                  "step and its head a sample)";
    }
    if (!r.base_loras.empty()) {
      out->note += fmt("; {} frozen adapter(s) beneath ({} MB)",
                       r.base_loras.size(), base >> 20)();
    }
    return true;
  }

  bool
  split_dims(const std::string& root, const std::string& model_type,
             const RunShape& r, int* n_heads, int* n_ffn,
             std::string* err) const override
  {
    if (variant_(root) == V::kPh3) {
      *err = "P-H3-Image does not train split across hosts yet";
      return false;
    }
    M::Config cfg;
    if (!M::config_from_json(dit_of_(root, model_type, r), cfg, err,
                             partition_(model_type))) {
      return false;
    }
    *n_heads = cfg.n_heads;
    *n_ffn = cfg.ffn;
    return true;
  }

  std::unique_ptr<LoraTrainer>
  load(const LoadArgs& a, std::string* err) const override
  {
    MiniMaxH3LoraTrainer::LoadArgs la;
    la.mc = a.mc;
    la.ws = a.ws;
    la.stream_blocks = a.stream_blocks;
    FlexData o = a.plan->opts;
    auto oo = o.as_object();
    la.partition = std::string(oo.at("partition").as_string(""));
    if (!M::config_from_json(
            a.plan->dit_dir, la.cfg, err,
            std::string(oo.at("dit_partition").as_string("")))) {
      return nullptr;
    }
    la.variant = (V)oo.at("variant").as_int(0);
    if (a.run != nullptr) {
      for (const std::string& l : a.run->base_loras) {
        M::LoraSpec sp;
        sp.path = l;
        la.base_loras.push_back(sp);
      }
    }
    if (la.variant == V::kPh3) {
      ph3::Manifest man;
      std::string merr;
      if (!ph3::read_manifest(a.root, &man, &merr)) {
        if (err != nullptr) { *err = merr; }
        return nullptr;
      }
      la.ph3_sigmas = man.sample_sigmas;
      la.partition = "fl2va";
    }
    if (la.variant == V::kH3i) {
      h3i::Manifest man;
      std::string merr;
      if (!h3i::read_manifest(a.root, &man, &merr)) {
        if (err != nullptr) { *err = merr; }
        return nullptr;
      }
      la.h3i_sigmas = man.sigmas;
      la.heads_ws = a.aux_ws;
      la.partition.clear();
    }
    la.cfg.prune_adaln = oo.at("prune_adaln").as_bool(false);
    // A rank's share (RunShape::tp): the load narrows the model to it.
    if (a.run != nullptr && a.run->tp.active()) { la.cfg.tp = a.run->tp; }
    // int8 matrix-core GEMMs in the tap forward, and the backward's dX
    // GEMMs on the same route (MiniMaxH3Trainer::set_i8_backward).
    la.cfg.i8_gemm = a.run != nullptr && a.run->i8;
    if (a.run != nullptr && a.run->max_grid_h > 0) {
      la.floor_seq = MiniMaxH3LoraTrainer::sequence_rows(
          la.cfg, a.run->max_frames, a.run->max_grid_h, a.run->max_grid_w,
          a.run->max_text, a.run->max_anchors, a.run->max_cond_rows);
      la.floor_text = a.run->max_text;
    }
    return MiniMaxH3LoraTrainer::load(la, err);
  }
};

}  // namespace

const std::vector<const LoraFamily*>&
lora_families()
{
  static const QwenImage21Family kQi21;
  static const MiniMaxH3Family kH3;
  static const std::vector<const LoraFamily*> kAll = {&kQi21, &kH3};
  return kAll;
}

}  // namespace train
}  // namespace genai
}  // namespace vpipe
