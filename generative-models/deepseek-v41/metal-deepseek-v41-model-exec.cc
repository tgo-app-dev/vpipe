#include "generative-models/deepseek-v41/metal-deepseek-v41-model-exec.h"
#include "generative-models/deepseek-v41/metal-deepseek-v41-vision.h"

#include "generative-models/generative-model-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/lm-run-memory.h"
#include "generative-models/shared/stream-sizing.h"
#include "generative-models/sampler.h"
#include "generative-models/weight-set.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string_view>

namespace vpipe::genai {

MetalDeepseekV41ModelExec::MetalDeepseekV41ModelExec(
    const std::string& model_dir, metal_compute::MetalCompute* mc,
    const MetalDeepseekV41Model::Options& opt,
    const SessionContextIntf* session)
    : _session(session)
{
  std::string err;
  auto ws = open_weight_set(model_dir, session);
  if (ws == nullptr) {
    if (session != nullptr) {
      session->warn(fmt("DeepSeek-V4.1: cannot open '{}'", model_dir));
    }
    return;
  }
  _model = MetalDeepseekV41Model::load(ws, mc, opt, &err);
  if (_model == nullptr && session != nullptr) {
    session->warn(fmt("DeepSeek-V4.1: {}", err));
  }
}

bool
MetalDeepseekV41ModelExec::expert_bytes(const std::string& model_dir,
                                        std::size_t* experts,
                                        std::size_t* rest,
                                        std::size_t* vision,
                                        std::size_t* disk)
{
  auto w = MetalLlamaWeights::open_model(model_dir);
  if (!w.has_value()) { return false; }
  std::size_t e = 0, r = 0, v = 0, d = 0;
  const bool keep = MetalDeepseekV41Model::keeps_fp8_projections();
  // An expert cache's slot holds any layer's expert, each of its weights
  // (w1, w2, w3) as wide as the widest layer's -- one size for every layer
  // of a pack that mixes widths (oQ2.5e) -- so every expert resident is
  // the layers times the widest of each weight: [weight][layer] bytes.
  std::vector<std::size_t> per[3];
  // An oQ pack's affine module: codes, a bf16 scale and bias per group.
  // Most are kept as stored; the small ones a decode reads as rows (the
  // compressor, the indexer's weights_proj and wk) are widened at load,
  // as is every one where widened projections were asked for -- at gs 64,
  // the packs' group, K = groups * 64 values a row.
  auto affine = [&](const std::string& n) {
    const std::size_t dot = n.rfind('.');
    return dot != std::string::npos &&
           w->info(n.substr(0, dot) + ".biases") != nullptr;
  };
  for (const std::string& nm : w->tensor_names()) {
    const auto* ti = w->info(nm);
    if (ti == nullptr) { continue; }
    d += ti->nbytes;
    // the release's names or an MLX pack's (under language_model.)
    const std::string n = nm.rfind("language_model.", 0) == 0
                              ? nm.substr(15) : nm;
    if (n.find(".ffn.experts.") != std::string::npos &&
        n.rfind("layers.", 0) == 0) {
      e += ti->nbytes;
      const std::size_t L = (std::size_t)std::atoi(n.c_str() + 7);
      const std::size_t at = n.find(".ffn.experts.");
      const std::string rest_n = n.substr(at + 13);
      // "w1.weight" (MLX's stacked) or "<e>.w1.weight" (the release's)
      const std::size_t wpos = rest_n.find('w');
      const int k = wpos != std::string::npos && wpos + 1 < rest_n.size()
                        ? rest_n[wpos + 1] - '1' : -1;
      if (k >= 0 && k < 3) {
        if (per[k].size() <= L) { per[k].resize(L + 1, 0); }
        per[k][L] += ti->nbytes;
      }
    } else if (n.rfind("vision.", 0) == 0 || n.rfind("aligner.", 0) == 0 ||
               n.rfind("image_", 0) == 0) {
      // the tower holds every tensor in the compute dtype
      std::size_t el = 1;
      for (std::int64_t d : ti->shape) { el *= (std::size_t)d; }
      v += el * 2;
    } else if (n.find(".engram.embed.") == std::string::npos &&
               n.rfind("mtp.", 0) != 0) {
      const bool small = n.find(".attn.compressor.") != std::string::npos ||
                         n.find(".indexer.weights_proj.") !=
                             std::string::npos ||
                         n.find(".indexer.wk.") != std::string::npos;
      if (affine(nm)) {
        const bool widen = !keep || small;
        if (ti->dtype == "U32") {
          const std::size_t dot = nm.rfind('.');
          const auto* si = w->info(nm.substr(0, dot) + ".scales");
          r += !widen ? ti->nbytes
               : si == nullptr || si->shape.size() != 2
                   ? 4 * ti->nbytes
                   : (std::size_t)si->shape[0] * si->shape[1] * 64 * 2;
        } else {
          r += widen ? 0 : ti->nbytes;     // its scales and biases
        }
        continue;
      }
      // FP8 codes (an MLX pack's in U32 words) are kept with a scale per
      // 32 of a row (the release's 32x32 block scales spread to their
      // rows); widened (the A/B), two bytes a code and no scales.
      const bool code = ti->dtype == "F8_E4M3" || ti->dtype == "U32";
      const bool scale = ti->dtype == "F8_E8M0" || ti->dtype == "U8";
      if (code) {
        r += keep ? ti->nbytes : 2 * ti->nbytes;
      } else if (scale) {
        r += !keep ? 0 : ti->dtype == "U8" ? ti->nbytes : 32 * ti->nbytes;
      } else {
        r += ti->nbytes;
      }
    }
  }
  if (!per[0].empty() && per[0].size() == per[1].size() &&
      per[1].size() == per[2].size()) {
    std::size_t widest = 0;
    for (const auto& p : per) {
      widest += *std::max_element(p.begin(), p.end());
    }
    e = std::max(e, widest * per[0].size());
  }
  if (experts != nullptr) { *experts = e; }
  if (rest != nullptr) { *rest = r; }
  if (vision != nullptr) { *vision = v; }
  if (disk != nullptr) { *disk = d; }
  return true;
}

std::size_t
MetalDeepseekV41ModelExec::lm_expert_cache_bytes(
    const SessionContextIntf* session, const std::string& model_dir,
    std::size_t run, const metal_compute::MetalCompute* mc, bool dspark,
    std::size_t limit, std::size_t cache)
{
  if (const char* e = std::getenv("VPIPE_DSV41_EXPERT_CACHE_MB")) {
    return (std::size_t)std::max(0LL, std::atoll(e)) << 20;
  }
  const std::size_t ram = stream_physical_ram();
  std::size_t experts = 0, rest = 0, vision = 0, disk = 0;
  if (ram == 0 ||
      !expert_bytes(model_dir, &experts, &rest, &vision, &disk) ||
      experts == 0) {
    return 0;
  }
  // the vision tower loads beside the model, its encode's scratch at the
  // largest picture with it
  DeepseekV41Config cfg;
  std::string cerr;
  if (vision > 0 && DeepseekV41Config::from_dir(model_dir, &cfg, &cerr)) {
    rest += vision + MetalDeepseekV41Vision::encode_scratch_bytes(cfg);
  }
  // and the DSpark head, held whole when loaded
  if (dspark) { rest += MetalDeepseekV41Model::dspark_bytes(model_dir); }
  std::size_t others = 0;
  auto* mgr = session != nullptr && session->services() != nullptr
                  ? session->services()->generative_model_manager()
                  : nullptr;
  if (mgr != nullptr) {
    others = mgr->phase_footprint(std::string());
    // without this checkpoint -- what the plan counts for it: a launch's
    // claim books it at its size on disk (the n-gram tables and DSpark
    // stages with it), or under a host's limit at what it will hold
    others -= std::min(others, mgr->planned_weight_bytes(model_dir));
    // the plan's activation scratch, but this model's own run (`run`,
    // which a stage's plan books under lm_run_label)
    const std::size_t sc = mgr->scratch_bytes(std::string());
    others += sc - std::min(sc, mgr->declared_scratch(
                                    lm_run_label(model_dir)));
  }
  constexpr std::size_t kHeadroom = (std::size_t)8 << 30;
  std::size_t budget = ram > others + run + kHeadroom
                           ? ram - others - run - kHeadroom
                           : 0;
  // The GPU's working set, 15% held back for the transient buffers a
  // forward and its peers allocate.
  if (mc != nullptr) {
    const std::size_t rec = mc->memory_budget().recommended;
    if (rec > 0) {
      const std::size_t cap = rec / 100 * 85;
      const std::size_t gpu = cap > others + run ? cap - others - run : 0;
      budget = std::min(budget, gpu);
    }
  }
  // The most the host lets it hold, the run in it.
  if (limit > 0) { budget = std::min(budget, limit > run ? limit - run : 0); }
  // The cache the host asks for: the rest and that, at most.
  if (cache > 0) { budget = std::min(budget, rest + cache); }
  if (rest + experts <= budget) { return 0; }
  // Whatever room is left; the cache itself never holds fewer than a
  // layer's experts resident plus a layer's transient (a prefill step may
  // route to all of them), so a box with no room still runs, slowly.
  const std::size_t left = budget > rest ? budget - rest : 0;
  return std::max<std::size_t>(left, 1);
}

bool
MetalDeepseekV41ModelExec::dspark_pays(const SessionContextIntf* session,
                                       const std::string& model_dir,
                                       std::size_t run,
                                       const metal_compute::MetalCompute* mc,
                                       std::size_t limit, std::size_t cache)
{
  if (const char* e = std::getenv("VPIPE_DSV41_DSPARK")) {
    return std::atoi(e) != 0;
  }
  if (MetalDeepseekV41Model::dspark_bytes(model_dir) == 0) { return false; }
  return lm_expert_cache_bytes(session, model_dir, run, mc, true, limit,
                               cache) == 0;
}

std::int32_t
MetalDeepseekV41ModelExec::prefill_multimodal_ids(
    ContextId ctx, std::span<const std::int32_t> ids,
    const std::vector<float>& rows, int n_mm,
    const std::vector<int>& positions)
{
  if (_model == nullptr || n_mm != (int)positions.size() ||
      rows.size() != (std::size_t)n_mm * _model->config().dim) {
    return -1;
  }
  MetalDeepseekV41Model::Media m;
  m.positions = positions;
  m.rows = rows.data();
  return run_(ctx, ids, &m);
}

std::int32_t
MetalDeepseekV41ModelExec::run_(ContextId ctx,
                                std::span<const std::int32_t> ids,
                                const MetalDeepseekV41Model::Media* media)
{
  if (_model == nullptr) { return -1; }
  std::string err;
  if (!_model->forward(ctx.v, ids, &_logits, /*all_rows=*/false, &err,
                       media)) {
    if (_session != nullptr) {
      _session->warn(fmt("DeepSeek-V4.1: {}", err));
    }
    return -1;
  }
  for (std::int32_t s : _suppress) {
    if (s >= 0 && (std::size_t)s < _logits.size()) {
      _logits[(std::size_t)s] = -std::numeric_limits<float>::infinity();
    }
  }
  std::int32_t best = 0;
  for (std::size_t i = 1; i < _logits.size(); ++i) {
    if (_logits[i] > _logits[(std::size_t)best]) { best = (std::int32_t)i; }
  }
  return best;
}

std::int32_t
MetalDeepseekV41ModelExec::prefill(ContextId ctx,
                                   std::span<const std::int32_t> tokens)
{
  return run_(ctx, tokens);
}

std::int32_t
MetalDeepseekV41ModelExec::decode_one(ContextId ctx, std::int32_t id)
{
  return run_(ctx, std::span<const std::int32_t>(&id, 1));
}

// ---- speculative decode --------------------------------------------------

void*
MetalDeepseekV41ModelExec::query_extension(std::string_view id) noexcept
{
  if (id == spec_decode::kExtensionId) {
    return static_cast<SpecDecodeExt*>(this);
  }
  return ModelExec::query_extension(id);
}

std::string_view
MetalDeepseekV41ModelExec::drafter() const noexcept
{
  return _model != nullptr && _model->has_dspark() ? spec_decode::kDrafterMtp
                                                   : std::string_view();
}

bool
MetalDeepseekV41ModelExec::attach_drafter(const std::string& dir, int bits,
                                          std::string* err)
{
  (void)bits;
  if (dir.empty()) { return true; }
  if (err != nullptr) {
    *err = "DeepSeek-V4.1 drafts with its own DSpark head (mtp.*), not a "
           "separate drafter";
  }
  return false;
}

namespace {
// The prior for a verify row's cost, in plain steps, until one is measured
// (MEASURED at 128 GB on the M5 Ultra: 0.35-0.52 a row).
constexpr double kDraftRowCost = 0.5;
}  // namespace

double
MetalDeepseekV41ModelExec::verify_cost_(int rows) const
{
  // a plain step's ms times the r-row verify's measured ratio to it; an
  // unmeasured ratio interpolated between measured ones (r = 1 is 1), and
  // past the last at its slope, or the prior's (half a step a row) with
  // one point
  const DraftChoice& dc = _choice;
  const double v1 = dc.nv[1] > 0 ? dc.v[1] : 1.0;
  const int n = (int)dc.v.size();
  const auto g = [&](int r) { return r == 1 ? 1.0 : dc.v[(std::size_t)r]; };
  const auto known = [&](int r) {
    return r == 1 || (r < n && dc.nv[(std::size_t)r] > 0);
  };
  if (rows <= 1) { return v1; }
  if (known(rows)) { return v1 * g(rows); }
  int lo = rows - 1, hi = rows + 1;
  while (!known(lo)) { --lo; }
  while (hi < n && !known(hi)) { ++hi; }
  if (hi < n) {
    return v1 * (g(lo) + (g(hi) - g(lo)) * (rows - lo) / (double)(hi - lo));
  }
  int lo2 = lo - 1;
  while (lo2 >= 1 && !known(lo2)) { --lo2; }
  const double slope =
      lo2 >= 1 ? (g(lo) - g(lo2)) / (double)(lo - lo2) : kDraftRowCost;
  return v1 * (g(lo) + std::max(slope, 0.0) * (rows - lo));
}

int
MetalDeepseekV41ModelExec::verify_count_(const std::vector<float>& conf,
                                         int k_max, double* rate,
                                         double draft_ms) const
{
  // E(k) = 1 + sum_j P(drafts 0..j-1 kept): the tokens a verify of k
  // drafts yields, each draft's chance its confidence bin's kept share at
  // its row (a prior of two rounds at the bin's own sigmoid). The draft is
  // paid by now, so the rate compares the verifies alone -- k = 0 (the
  // pending token alone) included.
  const DraftChoice& dc = _choice;
  int best = 0;
  double e = 1.0, p = 1.0, best_e = 1.0;
  const double t0 = verify_cost_(1);
  double best_rate = t0 > 0.0 ? 1.0 / t0 : 0.0;
  for (int k = 1; k <= k_max; ++k) {
    const double x = conf[(std::size_t)k - 1];
    const int bin = std::clamp((int)(10.0 / (1.0 + std::exp(-x))), 0, 9);
    const std::size_t u = (std::size_t)(k - 1) * 10 + (std::size_t)bin;
    const double q = (dc.kept[u] + 2.0 * (bin + 0.5) / 10.0) /
                     (dc.reached[u] + 2.0);
    p *= q;
    e += p;
    const double t = verify_cost_(k + 1);
    const double rk = t > 0.0 ? e / t : 0.0;
    if (rk > best_rate) {
      best_rate = rk;
      best = k;
      best_e = e;
    }
  }
  // the round's rate with its draft counted: what drafting is worth
  if (rate != nullptr) {
    const double t = (draft_ms >= 0.0 ? draft_ms : dc.d) +
                     verify_cost_(best + 1);
    *rate = t > 0.0 ? best_e / t : 0.0;
  }
  return best;
}

double
MetalDeepseekV41ModelExec::kept_share_(float conf, int at) const
{
  // the calibration's kept share for a draft at position `at` and its
  // confidence's bin (a prior of two rounds at the bin's own sigmoid), the
  // position past the measured ones the last's
  const DraftChoice& dc = _choice;
  const int bin = std::clamp((int)(10.0 / (1.0 + std::exp(-conf))), 0, 9);
  const int rows = (int)dc.kept.size() / 10;
  const std::size_t u =
      (std::size_t)std::clamp(at, 0, std::max(rows - 1, 0)) * 10 +
      (std::size_t)bin;
  if (u >= dc.kept.size()) { return (bin + 0.5) / 10.0; }
  return (dc.kept[u] + 2.0 * (bin + 0.5) / 10.0) / (dc.reached[u] + 2.0);
}

bool
MetalDeepseekV41ModelExec::generate(const SpecDecodeRequest& rq,
                                    SpecDecodeResult* out)
{
  SpecDecodeResult r;
  if (_model == nullptr || !_model->has_dspark() || rq.first_token < 0 ||
      rq.max_tokens <= 0) {
    return false;
  }
  const DeepseekV41Config& c = _model->config();
  const int V = c.vocab, B = c.dspark_block;
  const int nd_max = rq.block > 1 ? std::min(B, rq.block - 1) : B;
  // The verify's picks: the argmax, or the host sampler drawing once for
  // each kept token in order -- what the serial loop draws.
  const GpuSamplerParams& g = rq.sampler;
  SamplerParams sp;
  sp.temperature = g.greedy ? 0.0f : g.temperature;
  sp.top_k = g.top_k;
  sp.top_p = g.top_p;
  sp.min_p = g.min_p;
  sp.repetition_penalty = g.repetition_penalty;
  sp.presence_penalty = g.presence_penalty;
  sp.seed = g.seed;
  Sampler smp(sp);
  if (!rq.prime.empty()) { smp.prime(rq.prime); }
  smp.prime(std::span<const std::int32_t>(&rq.first_token, 1));
  const bool greedy = g.greedy || smp.is_argmax();
  std::vector<float> row((std::size_t)V);
  auto pick = [&](const float* lg) -> std::int32_t {
    std::copy(lg, lg + V, row.begin());
    for (std::int32_t s : _suppress) {
      if (s >= 0 && s < V) {
        row[(std::size_t)s] = -std::numeric_limits<float>::infinity();
      }
    }
    if (!greedy) { return smp.sample(row); }
    std::int32_t best = 0;
    for (int v = 1; v < V; ++v) {
      if (row[(std::size_t)v] > row[(std::size_t)best]) { best = v; }
    }
    return best;
  };
  auto is_stop = [&](std::int32_t t) {
    return rq.is_stop != nullptr && *rq.is_stop && (*rq.is_stop)(t);
  };
  auto emit = [&](std::span<const std::int32_t> t) {
    return rq.on_tokens == nullptr || !*rq.on_tokens || (*rq.on_tokens)(t);
  };
  const MetalDeepseekV41Model::CtxId ctx = rq.ctx.v;
  std::int32_t pending = rq.first_token;
  std::string err;
  if (is_stop(pending)) {
    r.hit_stop = true;
    if (out != nullptr) { *out = std::move(r); }
    return true;
  }
  r.produced = 1;
  bool go = emit(std::span<const std::int32_t>(&pending, 1));
  std::vector<std::int32_t> drafts, rows, kept;
  std::vector<float> conf, lg;
  DraftStats& ds = _dstats;
  ds.reached.resize((std::size_t)B * 10);
  ds.kept.resize((std::size_t)B * 10);
  ds.verify_ms.resize((std::size_t)B + 2);
  ds.verifies.resize((std::size_t)B + 2);
  using Clk = std::chrono::steady_clock;
  auto ms = [](Clk::time_point a, Clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  // HOW MANY DRAFTS A ROUND VERIFIES (VPIPE_DSV41_DRAFTS=<n> pins it, the
  // block's 5 the whole block): every verified row costs the GPU a
  // row and, where the experts are cached, the misses its picks add, while
  // a draft's chance to be kept falls along the block. So a round verifies
  // the k (0 included) of most tokens a ms -- E(k) / verify(k + 1), E from
  // the confidence head's calibration on this run, the costs measured --
  // and drafts at all only while that rate with the draft counted beats a
  // plain decode step (verify(1)). One round in kExplore is a plain step
  // or, every other time, a row count in turn, so every cost stays
  // measured.
  DraftChoice& dc = _choice;
  dc.v.resize((std::size_t)B + 2);
  dc.nv.resize((std::size_t)B + 2);
  dc.reached.resize((std::size_t)B * 10);
  dc.kept.resize((std::size_t)B * 10);
  const char* dv = std::getenv("VPIPE_DSV41_DRAFTS");
  const int pin = dv != nullptr ? std::atoi(dv) : 0;
  // THE DRAFT (VPIPE_DSV41_MTP_DRAFT, or set_mtp_modes): DSpark's one pass
  // over [token, noise...] ("block", the default), or CHAINED ("chain"):
  // a pass a draft, each over [token, the drafts so far] as an MTP head
  // chains, the next draft the last row's. THE DEPTH
  // (VPIPE_DSV41_MTP_DEPTH): the drafts a round verifies by measured rate
  // ("measured", the default; below), or by acceptance alone as omlx's
  // Lightning MTP chooses it ("acceptance"): one past the drafts the
  // round before kept, from 1 to the block, starting at 3.
  bool chain = _mtp_chain, by_acc = _mtp_acceptance;
  if (const char* e = std::getenv("VPIPE_DSV41_MTP_DRAFT")) {
    chain = std::string_view(e) == "chain";
  }
  if (const char* e = std::getenv("VPIPE_DSV41_MTP_DEPTH")) {
    by_acc = std::string_view(e) == "acceptance";
  }
  const bool choose = pin <= 0 && !by_acc;
  const int nd_round = pin > 0 ? std::min(nd_max, pin) : nd_max;
  // moving averages over ~8 samples: a cost moves as the cache warms (a
  // 6-row verify 314 ms cold, 160 warm), and a long memory keeps the cold
  // figures -- which is also why a verify is kept as its RATIO to a plain
  // step, the two moving together
  auto ema = [](double* m, int* n, double x) {
    *m = *n == 0 ? x : *m + (x - *m) / std::min(*n + 1, 8);
    ++*n;
  };
  while (go && r.produced < rq.max_tokens) {
    const int pos0 = _model->seq_len(ctx);
    // drafts this round: no more than the budget can keep
    int k = std::min(nd_round, rq.max_tokens - r.produced - 1);
    if (by_acc && pin <= 0) { k = std::min(k, _acc_depth); }
    int explore = -1;
    if (choose && k > 0) {
      ++dc.rounds;
      if (dc.rounds % (std::uint64_t)DraftChoice::kHalve == 0) {
        for (double& x : dc.reached) { x *= 0.5; }
        for (double& x : dc.kept) { x *= 0.5; }
      }
      const std::uint64_t ex = dc.rounds / DraftChoice::kExplore;
      if (dc.nv[1] == 0) {
        explore = 0;                   // a plain step's cost first
      } else if (dc.rounds % (std::uint64_t)DraftChoice::kExplore == 0) {
        // a plain step every other time, then each count in turn
        explore = ex % 2 == 0 ? 0 : 1 + (int)((ex / 2) % (std::uint64_t)k);
      } else if (dc.nd > 0 && dc.spark > 0.0 &&
                 1.0 / verify_cost_(1) > dc.spark) {
        k = 0;                         // a plain step beats drafting
      }
      if (explore == 0) { k = 0; }
    }
    drafts.clear();
    const auto t0 = Clk::now();
    if (k > 0 && chain) {
      // a pass a draft: under the measured rule another pass runs while
      // its expected tokens a ms (the calibration's kept share at the
      // last draft's confidence, one more pass and verify row paid) beat
      // stopping
      if (explore > 0) { k = std::min(k, explore); }
      conf.clear();
      std::vector<std::int32_t> one;
      std::vector<float> c1;
      bool bad = false;
      double e = 1.0, pk = 1.0;
      for (int j = 0; j < k; ++j) {
        const auto tp = Clk::now();
        if (!_model->dspark_draft(ctx, pending, &one, &c1, nullptr, &err,
                                  &drafts) || one.empty() || c1.empty()) {
          bad = true;
          break;
        }
        drafts.push_back(one[0]);
        conf.push_back(c1[0]);
        const double pms = ms(tp, Clk::now());
        ++ds.drafts;
        if (choose && ds.drafts > 2) { ema(&dc.d, &dc.nd, pms); }
        if (!choose || explore > 0 || j + 1 >= k || dc.nd == 0 ||
            dc.nv[1] == 0) {
          continue;
        }
        const double q = kept_share_(conf[(std::size_t)j], j);
        pk *= q;
        e += pk;
        const int d1 = j + 1;
        const double now_t = d1 * dc.d + verify_cost_(d1 + 1);
        const double next_t = (d1 + 1) * dc.d + verify_cost_(d1 + 2);
        if ((e + pk * q) / next_t <= e / now_t) { break; }
      }
      if (bad) { break; }
      if (_draft_override) { _draft_override(pending, &drafts); }
      const int passes = (int)drafts.size();
      ds.draft_ms += ms(t0, Clk::now());
      k = passes;
      if (choose) {
        double rate = 0.0;
        const int best = verify_count_(conf, k, &rate, passes * dc.d);
        if (dc.nd > 0 && dc.nv[1] > 0) {
          dc.spark = dc.spark == 0.0 ? rate
                                     : dc.spark + (rate - dc.spark) / 8.0;
        }
        k = explore > 0 ? std::min(k, explore) : best;
      }
      drafts.resize((std::size_t)k);
    } else if (k > 0) {
      if (!_model->dspark_draft(ctx, pending, &drafts, &conf, nullptr,
                                &err)) {
        break;
      }
      if (_draft_override) { _draft_override(pending, &drafts); }
      const double dms = ms(t0, Clk::now());
      ds.draft_ms += dms;
      ++ds.drafts;
      if (choose) {
        // the first drafts build their pipelines: not what a draft costs
        if (ds.drafts > 2) { ema(&dc.d, &dc.nd, dms); }
        // what drafting is worth, by this round's confidence
        double rate = 0.0;
        const int best = verify_count_(conf, k, &rate);
        if (dc.nd > 0 && dc.nv[1] > 0) {
          dc.spark = dc.spark == 0.0 ? rate
                                     : dc.spark + (rate - dc.spark) / 8.0;
        }
        k = explore > 0 ? std::min(k, explore) : best;
      }
      drafts.resize((std::size_t)k);
    }
    rows.assign(1, pending);
    rows.insert(rows.end(), drafts.begin(), drafts.end());
    const auto t1 = Clk::now();
    if (!_model->forward(ctx, rows, &lg, /*all_rows=*/true, &err)) { break; }
    const double vms = ms(t1, Clk::now());
    ds.verify_ms[rows.size()] += vms;
    ++ds.verifies[rows.size()];
    if (choose && rows.size() == 1) {
      ema(&dc.v[1], &dc.nv[1], vms);
    } else if (choose && dc.nv[1] > 0) {
      ema(&dc.v[rows.size()], &dc.nv[rows.size()], vms / dc.v[1]);
    }
    ++r.rounds;
    r.drafted += k;
    // row j predicts the token after rows[j]: kept while it is the draft
    kept.clear();
    int j = 0;
    bool stop = false;
    for (; j <= k; ++j) {
      const std::int32_t y = pick(lg.data() + (std::size_t)j * V);
      if (is_stop(y)) { stop = true; break; }
      kept.push_back(y);
      if (j < k && y == drafts[(std::size_t)j]) { continue; }
      break;
    }
    if (!stop && k > 0 && conf.size() >= (std::size_t)k) {
      for (int i = 0; i < k && i <= j; ++i) {
        const double x = conf[(std::size_t)i];
        const int bin = std::clamp((int)(10.0 / (1.0 + std::exp(-x))), 0, 9);
        const std::size_t u = (std::size_t)i * 10 + (std::size_t)bin;
        ++ds.reached[u];
        dc.reached[u] += 1.0;
        if (i < j) {
          ++ds.kept[u];
          dc.kept[u] += 1.0;
        }
      }
    }
    // rows 0..j consumed and kept; a draft row past them never ran
    const int keep_rows = std::min(j, k) + 1;
    if (!_model->truncate(ctx, pos0 + keep_rows)) {
      err = "truncate";
      break;
    }
    r.accepted += stop ? j : (long)kept.size() - 1;
    r.produced += (int)kept.size();
    if (by_acc && k > 0 && !stop) {
      _acc_depth = std::clamp(std::min(j, k) + 1, 1, B);
    }
    if (!kept.empty()) {
      go = emit(kept);
      pending = kept.back();
    }
    if (stop) {
      r.hit_stop = true;
      break;
    }
  }
  if (!err.empty() && _session != nullptr) {
    _session->warn(fmt("DeepSeek-V4.1 speculative decode: {}", err));
  }
  const bool ok = r.rounds > 0 || r.produced > 0;
  if (out != nullptr) { *out = std::move(r); }
  return ok;
}

bool
MetalDeepseekV41ModelExec::mtp_generate(
    ContextId ctx, std::int32_t first_token, int max_tokens, int rope_first,
    const GpuSamplerParams& sp,
    const std::function<bool(std::int32_t)>& is_stop,
    const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
    int* produced, bool* hit_stop)
{
  SpecDecodeRequest rq;
  rq.ctx = ctx;
  rq.first_token = first_token;
  rq.max_tokens = max_tokens;
  rq.rope_first = rope_first;
  rq.sampler = sp;
  rq.is_stop = is_stop ? &is_stop : nullptr;
  rq.on_tokens = on_tokens ? &on_tokens : nullptr;
  SpecDecodeResult r;
  const bool ok = generate(rq, &r);
  if (produced != nullptr) { *produced = r.produced; }
  if (hit_stop != nullptr) { *hit_stop = r.hit_stop; }
  return ok;
}

}  // namespace vpipe::genai
