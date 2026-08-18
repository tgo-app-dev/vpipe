#include "stages/model-memory.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/generative-model-manager.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"

#include <sys/sysctl.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace vpipe {
namespace model_memory {

std::size_t
phys_ram()
{
  // VPIPE_RAM_LIMIT_MB caps what every memory decision in the tree BELIEVES the
  // box has. That makes a bounded-box configuration reproducible on a big
  // machine (the 16 GB streaming/unload path is otherwise only testable on a
  // 16 GB machine), and it doubles as a way to leave headroom for other work.
  // stream-pin.h honours the same variable so the pinned-block count agrees.
  if (const char* e = std::getenv("VPIPE_RAM_LIMIT_MB")) {
    const long long mb = std::atoll(e);
    if (mb > 0) { return (std::size_t)mb << 20; }
  }
  std::uint64_t mem = 0;
  std::size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) != 0) { return 0; }
  return (std::size_t)mem;
}

std::size_t
dir_weights_bytes(const std::string& dir)
{
  namespace fs = std::filesystem;
  if (dir.empty()) { return 0; }
  std::size_t total = 0;
  std::error_code ec;
  // A checkpoint named by FILE, not by directory (a Comfy-Org repack:
  // one .safetensors per component). Without this the iterator below
  // just fails and reports 0 -- and a 66 GB DiT that sizes as free is
  // worse than no accounting at all, because every peer then plans
  // against a box that does not exist.
  if (fs::is_regular_file(fs::path(dir), ec) && !ec) {
    if (fs::path(dir).extension() != ".safetensors") { return 0; }
    const std::uintmax_t n = fs::file_size(fs::path(dir), ec);
    return ec ? 0 : (std::size_t)n;
  }
  for (fs::recursive_directory_iterator it(dir, ec), end; it != end;
       it.increment(ec)) {
    if (it->is_regular_file(ec) && it->path().extension() == ".safetensors") {
      total += (std::size_t)it->file_size(ec);
    }
  }
  return total;
}

std::size_t
weight_footprint(const SessionContextIntf*       session,
                 const std::vector<std::string>& dirs,
                 std::string_view                phase)
{
  namespace fs = std::filesystem;
  auto* mgr = session != nullptr ? session->services()->generative_model_manager()
                                 : nullptr;
  // Everything already open, counted once per checkpoint by the manager
  // -- narrowed to what will be resident during `phase` when the caller
  // named one.
  std::size_t total =
      mgr != nullptr ? mgr->phase_footprint(std::string(phase)) : 0;

  // Then the declared directories the manager does NOT already hold --
  // the ones this graph is about to load. Deduped against each other
  // too, so a component named by two stages is added once.
  std::vector<std::string> seen;
  seen.reserve(dirs.size());
  for (const std::string& d : dirs) {
    if (d.empty()) { continue; }
    std::string key = d;
    std::error_code ec;
    const fs::path c = fs::canonical(fs::path(d), ec);
    if (!ec) { key = c.string(); }
    bool dup = false;
    for (const std::string& s2 : seen) {
      if (s2 == key) { dup = true; break; }
    }
    if (dup) { continue; }
    seen.push_back(key);
    // Already accounted for -- open, or declared before launch. Adding
    // its on-disk size here would count the same checkpoint twice.
    if (mgr != nullptr && mgr->accounts_for(d)) { continue; }
    total += dir_weights_bytes(d);
  }
  return total;
}

bool
bounded(const SessionContextIntf*       session,
        const std::vector<std::string>& dirs,
        std::size_t                     headroom)
{
  const std::size_t ram = phys_ram();
  if (ram == 0) { return false; }
  return ram < weight_footprint(session, dirs) + headroom;
}

StreamPlan
plan_streaming(const SessionContextIntf* session, const std::string& dit_dir,
               const std::string& enc_dir, std::size_t headroom,
               std::size_t dit_retires)
{
  StreamPlan p;
  const std::size_t ram = phys_ram();
  if (ram == 0) { return p; }              // unknown: keep the roomy path
  // Asked in the DENOISE phase: peers that will have let go before this
  // model needs the room are not what it has to coexist with. That has
  // to come from a DECLARATION rather than from an event at unload
  // time, because this decision is taken on the first conditioning beat
  // -- after the encoder has produced its output and before it has been
  // dropped, so an announcement at unload arrives one decision too late.
  //
  // MEASURED on a 64 GB box with the bf16 MiniMax-H3: 57 GB of
  // footprint, of which 20 GB was an encoder that no longer existed by
  // the first denoise step, and the verdict turned on 1 GB.
  p.footprint = weight_footprint(session, {dit_dir, enc_dir}, kPhaseDenoise);
  // OTHERS INCLUDES THIS DIT, AND THAT IS DELIBERATE.
  //
  // The field reads like it should not -- and it was "corrected" once to
  // subtract this checkpoint, which makes pin_frac large for any DiT
  // bigger than RAM - 5 GB. That is the wrong lever. A precomputed
  // fraction cannot sense the machine: another process holding wired
  // memory, a peer that has not loaded yet, a box that is busy. Sizing a
  // big STATIC prefix from it pins memory nothing can give back, and
  // MEASURED on a 16 GB box it took a run that used no swap at all to
  // 11 GB resident with 3 GB of swap moving continuously.
  //
  // The prefix is deliberately the CONSERVATIVE half of the policy. What
  // adapts is BlockResidency, which grows the resident set by measuring
  // -- admitting against the live budget, and shedding when it finds its
  // own pages have left RAM. Feedback is what belongs on a shared
  // machine; a number computed before the run is not.
  //
  // So `others` stays inclusive, pin_frac stays small on a bounded box,
  // and growth does the work. A model that wants more resident blocks
  // should make growth converge better -- see
  // BlockResidency::note_landscape_changed, which is how MiniMax-H3
  // recovers the ratchet after its AdaLN bake frees 13.2 GB.
  p.others    = weight_footprint(session, {enc_dir}, kPhaseDenoise);
  // What the phase actually removed, for the caller's log line. The
  // unphased view is the no-release worst case, so the difference is
  // exactly the bytes some peer promised to have dropped.
  {
    const std::size_t unphased = weight_footprint(session, {dit_dir, enc_dir});
    p.transient = unphased > p.footprint ? unphased - p.footprint : 0;
  }
  // Weights the DiT reads once and drops never have to coexist with
  // anything, so they are not what this decision is about. Clamped
  // rather than trusted: a caller that over-states cannot drive the
  // footprint below what remains.
  p.retires   = std::min(dit_retires, p.footprint);
  p.footprint -= p.retires;
  p.stream    = ram < p.footprint + headroom;
  // Record the decision where peers can see it. Deliberately here and
  // not at the five call sites: this is the single streaming rule for
  // every DiT family, so it is also the only place that cannot drift
  // out of step with what was actually decided. That makes this a
  // decision function rather than a pure query -- every caller acts on
  // what it returns.
  if (p.stream && session != nullptr &&
      session->services() != nullptr &&
      session->services()->generative_model_manager() != nullptr) {
    session->services()->generative_model_manager()->note_streaming(dit_dir);
  }
  // Pin as many leading blocks as fit beside everything else that stays
  // resident. 5 GB covers activation scratch plus the in-flight block
  // and its double-buffer margin; 0.60 is the ceiling stream-pin.h
  // budgets against.
  if (p.stream && ram > p.others + (5ull << 30)) {
    p.pin_frac = std::min(0.60,
                          double(ram - p.others - (5ull << 30)) / double(ram));
  }
  return p;
}

bool
peer_streams(const SessionContextIntf* session)
{
  if (session == nullptr || session->services() == nullptr) { return false; }
  const auto* mgr = session->services()->generative_model_manager();
  return mgr != nullptr && mgr->any_streaming();
}

UnloadPolicy
parse_unload_policy(const std::string& s, bool* bad)
{
  if (bad != nullptr) { *bad = false; }
  if (s.empty() || s == "auto") { return UnloadPolicy::kAuto; }
  // Legacy spellings first, since they are what existing pipeline JSON
  // says: "always" meant destroy and "never" meant keep, back when park
  // had no name.
  if (s == "always" || s == "destroy") { return UnloadPolicy::kDestroy; }
  if (s == "never"  || s == "keep")    { return UnloadPolicy::kKeep; }
  if (s == "park") { return UnloadPolicy::kPark; }
  if (bad != nullptr) { *bad = true; }
  return UnloadPolicy::kAuto;
}

const char*
unload_policy_name(UnloadPolicy p)
{
  switch (p) {
    case UnloadPolicy::kDestroy: return "destroy";
    case UnloadPolicy::kPark:    return "park";
    case UnloadPolicy::kKeep:    return "keep";
    default:                     return "auto";
  }
}

std::size_t
vae_decode_scratch_bytes(const std::string& root, int width, int height)
{
  namespace fs = std::filesystem;
  if (root.empty() || width <= 0 || height <= 0) { return 0; }
  std::ifstream in(fs::path(root) / "vae" / "config.json");
  if (!in) { return 0; }
  FlexData fd = FlexData::from_json(in);
  if (!fd.is_object()) { return 0; }
  auto o = fd.as_object();

  const std::string cls =
      o.contains("_class_name") && o.at("_class_name").is_string()
          ? std::string(FlexData(o.at("_class_name")).get_string())
          : std::string();

  // The Qwen-Image VAE spells its width `base_dim` and ships NO
  // block_out_channels at all -- every quantized Krea-2 and
  // Qwen-Image-Edit pack has the key absent. Read as block_out it fell
  // to the 128 default against a real 96, over-declaring 1.33x: the
  // safe direction, but not a number anybody computed, and it made an
  // "exact" claim for Krea-2 that was not.
  if (cls == "AutoencoderKLQwenImage") {
    int base = 96;                        // the stock default
    if (o.contains("base_dim")) {
      const int b = (int)FlexData(o.at("base_dim")).as_real(0.0);
      if (b > 0) { base = b; }
    }
    // im2col h*w*9*base*2, plus ~50% for the level's I/O.
    return (std::size_t)height * (std::size_t)width * (std::size_t)base * 27;
  }

  // Base channels: max(block_out[0], block_out[1]), the width the 2K
  // overflow fix settled on -- block_out[0] alone under-modelled the
  // peak 2x. 128 is the stock AutoencoderKL default when unreadable.
  int base = 128;
  if (o.contains("block_out_channels")) {
    FlexData bo = o.at("block_out_channels");
    auto arr = bo.as_array();          // owner `bo` outlives this view
    if (arr.size() >= 2) {
      const int b0 = (int)arr[0].as_real(0.0);
      const int b1 = (int)arr[1].as_real(0.0);
      if (b0 > 0 && b1 > 0) { base = std::max(b0, b1); }
    } else if (arr.size() >= 1) {
      const int b0 = (int)arr[0].as_real(0.0);
      if (b0 > 0) { base = b0; }
    }
  }

  // Multipliers on (h * w * base), each mirroring that VAE class's own
  // decode_peak_bytes:
  //   AutoencoderKLFlux2      ~7 full-res base buffers at 2 B/elt  -> 14
  //   AutoencoderKLQwenImage  im2col h*w*9*base*2, +50% for I/O     -> 27
  // Unknown gets 27: over-declaring an arena costs caution, under-
  // declaring reads as room that is not there.
  // Boogu ships the plain `AutoencoderKL`, and this tree decodes it
  // through MetalFlux2Vae -- the same code, so the same peak. Left at
  // the conservative 27 it over-declared Boogu 1.93x, which is the safe
  // direction but is still a number nobody computed.
  const std::size_t mult =
      (cls == "AutoencoderKLFlux2" || cls == "AutoencoderKL") ? 14 : 27;
  return (std::size_t)height * (std::size_t)width * (std::size_t)base * mult;
}

bool
resolve_idle_unload(std::size_t ram, std::size_t peers, std::size_t arena,
                    bool current)
{
  if (ram == 0) { return current; }        // unknown box: do not churn
  const std::size_t need = peers + arena;
  if (ram < need) { return true; }                    // does not fit
  if (ram >= need + arena / 8) { return false; }      // fits comfortably
  return current;                                     // inside the band
}

std::size_t
video_decode_scratch_bytes(int width, int height, int frames)
{
  if (width <= 0 || height <= 0 || frames <= 0) { return 0; }
  const std::size_t px =
      (std::size_t)height * (std::size_t)width * (std::size_t)frames;
  return px * 3 * 2      // the decode's own output, bf16
       + px * 3;         // the planar-U8 clip the stage buffers behind it
}

std::vector<ResourceClaim>
scratch_claims(std::string label, std::size_t bytes, std::string_view phase)
{
  if (label.empty() || bytes == 0) { return {}; }
  // The key carries both, because a ResourceClaim has no byte field and
  // the planner cannot derive an arena's size from its name the way it
  // derives a checkpoint's from its directory. Formatted here so no
  // stage ever spells the encoding itself.
  return {ResourceClaim{std::string(kScratchKind),
                        label + "|" + std::to_string(bytes),
                        std::string(phase)}};
}

std::size_t
scratch_footprint(const SessionContextIntf* session, std::string_view phase)
{
  auto* mgr = session != nullptr
                  ? session->services()->generative_model_manager()
                  : nullptr;
  return mgr != nullptr ? mgr->scratch_bytes(std::string(phase)) : 0;
}

std::vector<ResourceClaim>
weight_claims(std::vector<std::string> dirs)
{
  std::vector<ResourceClaim> out;
  out.reserve(dirs.size());
  for (std::string& d : dirs) {
    if (d.empty()) { continue; }
    out.push_back(ResourceClaim{std::string(kWeightsKind), std::move(d)});
  }
  return out;
}

std::vector<ResourceClaim>
weight_claims_in_phase(std::vector<std::string> dirs,
                       std::string_view         phase)
{
  std::vector<ResourceClaim> out;
  out.reserve(dirs.size());
  for (std::string& d : dirs) {
    if (d.empty()) { continue; }
    out.push_back(ResourceClaim{std::string(kWeightsKind), std::move(d),
                                std::string(phase)});
  }
  return out;
}

namespace {

// The planner behind kWeightsKind: every checkpoint the graph is about
// to open, on the manager's books before the first stage loads one.
//
// This is the whole of what PipelineRuntime used to do inline, and it
// is deliberately the only place that knows a claim key is a directory
// and that a directory has a size.
class ModelWeightPlanner final : public ResourcePlanner {
public:
  std::string_view
  kind() const noexcept override
  {
    return kWeightsKind;
  }

  void
  begin_plan(const SessionContextIntf* session) override
  {
    // Drop the previous launch's estimates so one run's declarations
    // never leak into the next. Unconditional -- a graph with no
    // weight claims at all still has to clear what the last one left.
    if (auto* mgr = manager(session)) { mgr->clear_declarations(); }
    _declared.store(0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(_mu);
    _pending.clear();
  }

  void
  claim(const SessionContextIntf* session, const std::string& dir,
        const std::string& phase) override
  {
    auto* mgr = manager(session);
    if (mgr == nullptr) { return; }
    const std::size_t b = dir_weights_bytes(dir);
    if (b == 0) { return; }        // absent component: nothing to hold
    // A phase on a DECLARATION is ignored, and said out loud. This pass
    // runs while the picture is still being assembled, so honouring one
    // here would make it visible to whichever stages are visited next
    // and invisible to the rest -- order-dependence in the phase built
    // to remove it. Phases belong in Stage::decide_resources.
    if (!phase.empty() && session != nullptr) {
      session->warn(fmt(
          "resource-plan: '{}' names phase '{}' in declare_resources, where "
          "it is ignored; move it to decide_resources", dir, phase));
    }
    mgr->declare_weights(dir, b);
    _declared.fetch_add(b, std::memory_order_relaxed);
  }

  void
  decide(const SessionContextIntf* session, const std::string& dir,
         const std::string& phase) override
  {
    if (phase.empty()) { return; }
    // An unrecognised phase is dropped, and says so. Silently honouring
    // it would put the claim in a phase of its own, counted apart from
    // the claims it really coexists with -- an UNDER-count, which is the
    // direction that thrashes. See kPhaseCondition.
    if (phase != kPhaseCondition && phase != kPhaseDenoise &&
        phase != kPhaseDecode) {
      if (session != nullptr) {
        session->warn(fmt(
            "resource-plan: '{}' decides unknown phase '{}'; counting it as "
            "resident for the whole run", dir, phase));
      }
      return;
    }
    // BUFFERED, not applied. Applying it here would let this stage's
    // refinement change the box the NEXT stage's decide_resources sizes
    // against, which is exactly the ordering effect this pass exists to
    // remove. end_plan applies them together.
    std::lock_guard<std::mutex> lk(_mu);
    _pending.push_back({dir, phase});
  }

  void
  end_plan(const SessionContextIntf* session) override
  {
    auto* mgr = manager(session);
    // Apply the buffered refinements, now that every stage has both
    // declared and decided against the same complete picture.
    std::size_t phased = 0;
    {
      std::vector<Pending> pending;
      {
        std::lock_guard<std::mutex> lk(_mu);
        pending.swap(_pending);
      }
      for (const Pending& p : pending) {
        if (mgr == nullptr) { break; }
        mgr->set_declaration_phase(p.dir, p.phase);
        phased += dir_weights_bytes(p.dir);
      }
    }

    const std::size_t d = _declared.load(std::memory_order_relaxed);
    if (d == 0 || session == nullptr) { return; }
    // `d` sums CLAIMS and a checkpoint two stages name is claimed
    // twice, so it is always the larger, cruder number. What the box
    // has to hold is the manager's deduped view.
    const std::size_t total = mgr != nullptr ? mgr->resident_weight_bytes() : d;
    session->log_debug(fmt(
        "resource-plan: {} MB of model weights declared before init ({} MB "
        "of claims, before shared checkpoints are counted once)",
        total >> 20, d >> 20));

    if (phased == 0) { return; }
    const std::size_t peak = mgr != nullptr ? mgr->phase_footprint({}) : total;
    // The peak differs from the total only once TWO phases are
    // declared -- with one, its maximum and its sum are the same
    // number. Said plainly rather than dressed up, because the value of
    // a single phase is elsewhere: peers in a DIFFERENT phase stop
    // counting these bytes at all, which is what plan_streaming asks
    // for and what no total can express.
    session->log_debug(fmt(
        "resource-plan: {} MB of that is phase-limited; peak across phases "
        "{} MB", phased >> 20, peak >> 20));
  }

private:
  static genai::GenerativeModelManager*
  manager(const SessionContextIntf* session)
  {
    return session != nullptr ? session->services()->generative_model_manager()
                              : nullptr;
  }

  // Atomic only because the planner is a process-wide singleton and two
  // pipelines can launch at once. It feeds a debug line, so a garbled
  // total across concurrent launches is acceptable; a data race is not.
  std::atomic<std::size_t> _declared{0};

  // Refinements from the decide pass, held until end_plan applies them
  // together. See decide() for why they cannot be applied on arrival.
  struct Pending { std::string dir; std::string phase; };
  mutable std::mutex   _mu;
  std::vector<Pending> _pending;
};

}

namespace {

// The planner behind kScratchKind. Unlike a checkpoint, an arena has no
// size on disk to look up, so the claim carries it: "<label>|<bytes>".
class ScratchPlanner final : public ResourcePlanner {
public:
  std::string_view
  kind() const noexcept override
  {
    return kScratchKind;
  }

  void
  begin_plan(const SessionContextIntf* session) override
  {
    if (auto* mgr = manager(session)) { mgr->clear_scratch(); }
    _total.store(0, std::memory_order_relaxed);
    _arenas.store(0, std::memory_order_relaxed);
  }

  void
  claim(const SessionContextIntf* session, const std::string& key,
        const std::string& phase) override
  {
    auto* mgr = manager(session);
    if (mgr == nullptr) { return; }
    const std::size_t bar = key.rfind('|');
    if (bar == std::string::npos) {
      if (session != nullptr) {
        session->warn(fmt(
            "resource-plan: scratch claim '{}' is not '<label>|<bytes>'; "
            "ignored", key));
      }
      return;
    }
    std::size_t bytes = 0;
    try {
      bytes = (std::size_t)std::stoull(key.substr(bar + 1));
    } catch (const std::exception&) {
      bytes = 0;
    }
    if (bytes == 0) { return; }
    mgr->declare_scratch(key.substr(0, bar), bytes, phase);
    _total.fetch_add(bytes, std::memory_order_relaxed);
    _arenas.fetch_add(1, std::memory_order_relaxed);
  }

  void
  end_plan(const SessionContextIntf* session) override
  {
    const unsigned n = _arenas.load(std::memory_order_relaxed);
    if (n == 0 || session == nullptr) { return; }
    const std::size_t t = _total.load(std::memory_order_relaxed);
    auto* mgr = manager(session);
    const std::size_t peak = mgr != nullptr ? mgr->scratch_bytes({}) : t;
    // The COUNT, not just the bytes. An arena whose size is not knowable
    // until the first beat is declared at kUnknownArena and rounds to
    // 0 MB, and a line reading "0 MB declared" would say the opposite of
    // what happened.
    session->log_debug(fmt(
        "resource-plan: {} activation arena(s) declared, {} MB; peak across "
        "phases {} MB", n, t >> 20, peak >> 20));
  }

private:
  static genai::GenerativeModelManager*
  manager(const SessionContextIntf* session)
  {
    return session != nullptr ? session->services()->generative_model_manager()
                              : nullptr;
  }

  std::atomic<std::size_t> _total{0};
  std::atomic<unsigned>    _arenas{0};
};

}

VPIPE_REGISTER_RESOURCE_PLANNER(ScratchPlanner)
VPIPE_REGISTER_RESOURCE_PLANNER(ModelWeightPlanner)

}  // namespace model_memory
}  // namespace vpipe
