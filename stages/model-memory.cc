#include "stages/model-memory.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/shared/stream-pin.h"
#include "interfaces/session-context-intf.h"
#include "apple-silicon/metal-compute/metal-compute.h"
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

namespace {

// The running order. One list, so phase_order() and phase_count() cannot
// disagree about what exists or in what sequence.
constexpr std::string_view kPhasesInOrder[] = {
    kPhaseCondition, kPhaseDenoise, kPhaseDecodeAudio, kPhaseDecode,
};

}  // namespace

int
phase_order(std::string_view phase)
{
  for (int i = 0; i < (int)(sizeof(kPhasesInOrder) / sizeof(*kPhasesInOrder));
       ++i) {
    if (kPhasesInOrder[i] == phase) { return i; }
  }
  return -1;
}

int
phase_count()
{
  return (int)(sizeof(kPhasesInOrder) / sizeof(*kPhasesInOrder));
}

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
streaming_floor_bytes(const std::string&                   dir,
                      const std::vector<std::string_view>& stems,
                      std::string_view                     exclude)
{
  if (dir.empty()) { return 0; }
  auto wts = genai::MetalLlamaWeights::open_model(dir);
  if (!wts.has_value()) { return 0; }
  for (std::string_view stem : stems) {
    const std::size_t f = genai::stream_floor_bytes(*wts, stem, exclude);
    if (f > 0) { return f; }
  }
  return 0;
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
  // WHAT THE VERDICT WAS COMPUTED FROM, at debug level.
  //
  // The decision below is irreversible and the number behind it is a
  // sum over things this caller cannot see, so a wrong verdict is
  // otherwise a bare "STREAM" with nothing to check it against. That is
  // not hypothetical: an LTX-2.5 DiT read 43868 MB where the truth was
  // 18821 MB, because it named a peer's encoder that the peer had not
  // loaded -- and the only visible symptom was a model streaming a pack
  // that fitted four times over.
  //
  // `dirs` are what this caller is ABOUT TO HOLD. Naming a peer's
  // component here adds its on-disk size on top of whatever the peer
  // really declared, so the two figures below diverging by roughly a
  // checkpoint is the signature of that mistake.
  if (session != nullptr) {
    const auto* mgr = session->services() != nullptr
                          ? session->services()->generative_model_manager()
                          : nullptr;
    session->log_debug(fmt(
        "plan_streaming: {} MB RAM; this caller holds dit '{}' ({} MB) + "
        "enc '{}' ({} MB); footprint in '{}' {} MB, of which the manager "
        "already accounts {} MB; unphased {} MB",
        ram >> 20, dit_dir, dir_weights_bytes(dit_dir) >> 20,
        enc_dir, dir_weights_bytes(enc_dir) >> 20,
        kPhaseDenoise, p.footprint >> 20,
        mgr != nullptr
            ? mgr->phase_footprint(std::string(kPhaseDenoise)) >> 20 : 0,
        weight_footprint(session, {dit_dir, enc_dir}) >> 20));
  }
  p.retires   = std::min(dit_retires, p.footprint);
  p.footprint -= p.retires;
  // STREAM UNLESS THE MODEL IS SMALL AGAINST THE BOX.
  //
  // The two paths used to be near-equals, chosen by whether the
  // checkpoint fit with headroom to spare. They are not equals. Preload
  // is the fragile one: it is decided once, before the run, from an
  // estimate, and it cannot be walked back -- a model that declines to
  // stream and then holds more than predicted thrashes, and nothing
  // later can undo it. Streaming has a floor it can always fall to, and
  // BlockResidency grows the resident set back up as the box allows, so
  // on a machine with room the streamed path CONVERGES on what preload
  // would have held anyway. MEASURED on the M4 Pro 64 GB with the bf16
  // MiniMax-H3: growth reached 50 of 50 blocks and forwards 2-5 read
  // nothing at all.
  //
  // So preload is kept only where it is obviously safe -- a checkpoint
  // that is a small fraction of the box -- and streaming is the default
  // everywhere else. `headroom` still has to be there on top: a third of
  // a 16 GB box is 5.3 GB, which is small as a fraction and not small
  // beside everything else running.
  //
  // Against TOTAL ram rather than what is free right now, deliberately
  // and for the reason bounded() gives: every stage in a graph must
  // answer this the same way and it has to be reproducible, so the
  // figure cannot move underfoot between the first stage to ask and the
  // last.
  const bool small_against_box =
      p.footprint > 0 && p.footprint <= ram / 3 &&
      ram >= p.footprint + headroom;
  p.stream    = !small_against_box;
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

std::vector<ResourceClaim>
payload_claims(std::string label, std::size_t bytes,
               std::string_view first_phase, std::string_view last_phase)
{
  if (bytes == 0) { return {}; }
  ResourceClaim c;
  c.kind = std::string(kScratchKind);
  c.key = std::move(label) + "|" + std::to_string(bytes);
  c.phase = std::string(first_phase);
  c.last_phase = std::string(last_phase);
  return {std::move(c)};
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

ResourceClaim
weight_claim_streamable(std::string dir, std::size_t floor)
{
  ResourceClaim c;
  c.kind = std::string(kWeightsKind);
  c.key = std::move(dir);
  c.floor_bytes = floor;
  return c;
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
    if (auto* mgr = manager(session)) {
      mgr->clear_declarations();
      // The vocabulary lives here; the manager evaluates intervals over
      // it. Set every plan, so a manager reused across launches cannot
      // be left ordering by a stale list.
      std::vector<std::string> order;
      for (int i = 0; i < phase_count(); ++i) {
        order.push_back(std::string(
            i == 0   ? kPhaseCondition
            : i == 1 ? kPhaseDenoise
            : i == 2 ? kPhaseDecodeAudio
                     : kPhaseDecode));
      }
      mgr->set_phase_order(std::move(order));
    }
    report_landscape_(session);
    _declared.store(0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(_mu);
    _pending.clear();
  }

  // WHAT THE BOX LOOKS LIKE BEFORE ANYTHING LOADS.
  //
  // Every figure the planning phase reports afterwards is a claim about
  // what a graph will take; none of them says what was already there.
  // That is the missing half whenever a run turns out badly: a graph
  // sized correctly against a machine that was already holding 30 GB of
  // somebody else's memory reads, in the log, exactly like one sized
  // wrongly against an empty one.
  //
  // At INFO and once per launch. Deliberately a SNAPSHOT and labelled as
  // one -- these numbers move under the run, and the decisions below are
  // taken against total RAM precisely so they do not depend on them.
  static void
  report_landscape_(const SessionContextIntf* session)
  {
    if (session == nullptr || session->services() == nullptr) { return; }
    auto* mc = session->services()->metal_compute();
    if (mc == nullptr || !mc->valid()) { return; }
    const auto mb = mc->memory_budget();
    if (mb.total_physical == 0) { return; }
    const auto m = [](std::size_t b) { return b >> 20; };
    session->info(fmt(
        "memory landscape at launch: {} MB RAM -- {} MB idle, {} MB file "
        "cache, {} MB wired system-wide, {} MB compressed, {} MB swap; this "
        "process holds {} MB",
        m(mb.total_physical), m(mb.free_physical), m(mb.file_cache),
        m(mb.wired), m(mb.compressed), m(mb.swap_used),
        m(mb.self_footprint)));
    // The pool's own starting point, beside the ceiling the GPU sets. A
    // limit reported without the device maximum cannot be read: 48 GB is
    // roomy or already clamped depending on a number that is not there.
    if (auto* mgr = manager(session)) {
      const std::size_t lim = mgr->wired_pool_limit();
      if (lim == 0) {
        session->info(fmt(
            "memory landscape: the wired pool is OFF (wired_pool_pct=0), so "
            "nothing this run holds is protected from the compressor"));
      } else {
        const std::size_t dev = mgr->wired_pool_device_max();
        session->info(fmt(
            "memory landscape: wired pool {} MB{}, {} MB of it already in "
            "use{}",
            m(lim),
            dev > 0 ? fmt(" of {} MB the GPU can keep resident", m(dev))()
                    : std::string(),
            m(mgr->wired_pool_used()),
            mb.paging() ? " -- and the box is ALREADY paging, so growth "
                          "will be refused until that clears"
                        : ""));
      }
    }
  }

  void
  claim(const SessionContextIntf* session, const std::string& dir,
        const std::string& phase, const std::string& last_phase,
        std::size_t floor) override
  {
    auto* mgr = manager(session);
    if (mgr == nullptr) { return; }
    const std::size_t b = dir_weights_bytes(dir);
    if (b == 0) { return; }        // absent component: nothing to hold
    // A floor bigger than the checkpoint is a caller mistake, and taking
    // it would report a streamed total ABOVE the preloaded one. Clamp
    // rather than trust: this figure decides whether a graph is refused.
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
    mgr->declare_weights(dir, b, phase, last_phase,
                         floor > 0 && floor < b ? floor : 0);
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

  bool
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
    if (d == 0 || session == nullptr) { return true; }
    // `d` sums CLAIMS and a checkpoint two stages name is claimed
    // twice, so it is always the larger, cruder number. What the box
    // has to hold is the manager's deduped view.
    const std::size_t total = mgr != nullptr ? mgr->resident_weight_bytes() : d;
    const std::size_t fl =
        mgr != nullptr ? mgr->phase_footprint_floor({}) : 0;
    session->info(fmt(
        "resource-plan: {} MB of model weights declared before init ({} MB "
        "of claims, before shared checkpoints are counted once)",
        total >> 20, d >> 20));
    // BOTH numbers, because they answer different questions and a graph
    // can fit one and not the other. The preloaded figure is what the
    // box must hold if nothing streams; the streamed one is the sum of
    // what each component says it can be reduced to. A graph above the
    // first and below the second does not need refusing -- it needs its
    // streamable components told to stream.
    //
    // Both figures come from the manager's deduped, phase-narrowed view,
    // so they are comparable and their DIFFERENCE is what streaming is
    // worth on this graph. A floor summed beside a deduped total is not
    // comparable to it -- it reads larger, and the smaller of the two is
    // then always the total, which makes the floor do nothing at all.
    const std::size_t peak_full =
        mgr != nullptr ? mgr->phase_footprint({}) : d;
    if (fl > 0 && fl != peak_full) {
      session->info(fmt(
          "resource-plan: {} MB preloaded / {} MB if every streamable "
          "component streams", peak_full >> 20, fl >> 20));
    }

    if (phased != 0) { report_phases_(session, mgr, total, phased); }

    // ---- THE POOL CHECK, and the only place a graph is refused -------
    //
    // Refused only when the OPTIMISTIC reading does not fit: every
    // streamable component at its floor, weights counted once and
    // narrowed to their widest phase, plus the activation scratch. A
    // refusal has to be a fact, not an estimate -- this accounting has
    // been wrong before (a root-dir claim double-counting its own
    // subdirectories reported 230781 MB against a true 118452), and a
    // confident refusal computed from a number like that turns a working
    // run into a support ticket. Erring toward admitting leaves the
    // existing behaviour, which is merely slow.
    if (mgr == nullptr) { return true; }
    const std::size_t pool = mgr->wired_pool_limit();
    if (pool == 0) { return true; }        // wiring off: nothing to refuse
    // The PEAK -- the largest sum of everything alive at once, walking
    // the phases in order -- rather than weights-plus-widest-scratch.
    // The two differ by exactly the terms that outlive their producer:
    // the conditioning, the latent, the decoded audio and frames. On a
    // constrained box those are frequently the largest thing in the
    // moment they are live, and the old shape had nowhere to put them.
    std::vector<std::pair<std::string, std::size_t>> per_phase;
    const std::size_t need = mgr->phase_peak(&per_phase);
    if (need == 0) { return true; }
    std::string tight;
    std::size_t tight_b = 0;
    std::string breakdown;
    for (const auto& [p, b] : per_phase) {
      if (b > tight_b) { tight_b = b; tight = p; }
      if (!breakdown.empty()) { breakdown += ", "; }
      breakdown += fmt("{} {} MB", p, b >> 20)();
    }
    session->info(fmt(
        "resource-plan: peak {} MB in phase '{}' ({})",
        need >> 20, tight, breakdown));
    if (need <= pool) { return true; }

    // OVER THE POOL IS NOT OVER THE BOX, and only one of the two is a
    // problem the operator did not ask for.
    //
    // The pool is a ceiling on what this process WIRES, never on what it
    // may hold: bytes above it are pageable, which costs throughput and
    // not correctness. So a deliberately small pool on a large machine
    // -- `--wired-pool-mb 12000` on a 64 GB box -- is a setting being
    // honoured, and "it does not fit this machine" is then simply false.
    // MEASURED on the bf16 MiniMax-H3 FL2VA graph, which was told
    // exactly that and ran to completion.
    //
    // What genuinely cannot fit is a peak above believed RAM with every
    // streamable component already at its floor, and that is the only
    // reading this WARNS about. The pool shortfall is worth saying --
    // part of the run is exposed to the compressor -- so it is said at
    // INFO, where a number nobody has to act on belongs.
    //
    // 0 from phys_ram() is UNKNOWN, not small: the roomier reading is
    // the one that does not invent a refusal out of a failed sysctl.
    const std::size_t ram = phys_ram();
    const bool over_box = ram > 0 && need > ram;
    // Reported either way; refused only when asked to be, and then at
    // the POOL rather than the box -- that is what wired_pool_enforce
    // buys. An incompletely declared graph reads as too big, and a veto
    // on that basis blocks runs that work -- see
    // parse_wired_pool_enforce_config.
    const bool enforce = mgr->wired_pool_enforced();

    if (!over_box) {
      if (!enforce) {
        session->info(fmt(
            "resource-plan: peak {} MB in phase '{}' is beyond the {} MB "
            "wired pool, so {} MB of it stays pageable{}. {}",
            need >> 20, tight, pool >> 20, (need - pool) >> 20,
            ram > 0 ? fmt(" -- this machine's {} MB holds the graph",
                          ram >> 20)()
                    : std::string(),
            pool_advice_(mgr, need, ram)));
        return true;
      }
      // The opt-in veto, and it says which setting turned a report into
      // a refusal -- the same numbers reach INFO without it.
      session->error(fmt(
          "resource-plan: peak {} MB in phase '{}' is beyond the {} MB "
          "wired pool and wired_pool_enforce is on, so the graph is "
          "refused rather than run with {} MB of it pageable. {}",
          need >> 20, tight, pool >> 20, (need - pool) >> 20,
          pool_advice_(mgr, need, ram)));
      return false;
    }
    const VpipeFormat msg = fmt(
        "resource-plan: this graph needs at least {} MB resident at its "
        "peak -- phase '{}', with everything streamable at its floor -- "
        "and this machine has {} MB. It does not fit at any setting -- "
        "use a smaller model, a smaller geometry, or a quantized "
        "checkpoint.",
        need >> 20, tight, ram >> 20);
    if (enforce) { session->error(msg); } else { session->warn(msg); }
    return !enforce;
  }

private:
  // WHICH KNOB, and only when turning it would do something.
  //
  // An absolute `wired_pool_mb` REPLACES the percentage rather than
  // combining with it, so naming `wired_pool_pct` to an operator who
  // typed the absolute form recommends a setting that changes nothing.
  // Worse, a percentage recovered by dividing the pool by a pct it did
  // not come from invents a machine: 12000 MB "at 75%" reads as a 16 GB
  // box on a 64 GB one, and every figure derived from it is then wrong
  // in the same direction. Ask phys_ram(), never the pool.
  //
  // The device cap comes first because it outranks both forms -- the
  // pool is clamped to what the GPU can keep resident before anything is
  // wired, so an ask above it is not a setting, it is a wish.
  static std::string
  pool_advice_(genai::GenerativeModelManager* mgr, std::size_t need,
               std::size_t ram)
  {
    const std::size_t devmax = mgr->wired_pool_device_max();
    if (devmax > 0 && need > devmax) {
      return fmt("The GPU can keep {} MB resident, so no pool setting "
                 "protects all of it.", devmax >> 20)();
    }
    if (mgr->wired_pool_bytes() > 0) {
      return fmt("Set wired_pool_mb to {} or more to protect all of it.",
                 need >> 20)();
    }
    const int want = ram > 0 ? (int)((need * 100 + ram - 1) / ram) : 0;
    // Above 95 there is no such percentage -- saying "raise it to 723%"
    // is worse than saying nothing, because it reads like a setting.
    if (want > 0 && want <= 95) {
      return fmt("Set wired_pool_pct to {} or more to protect all of it.",
                 want)();
    }
    return "No wired_pool_pct covers all of it on this box.";
  }

  void
  report_phases_(const SessionContextIntf* session,
                 genai::GenerativeModelManager* mgr, std::size_t total,
                 std::size_t phased) const
  {
    const std::size_t peak = mgr != nullptr ? mgr->phase_footprint({}) : total;
    // The peak differs from the total only once TWO phases are
    // declared -- with one, its maximum and its sum are the same
    // number. Said plainly rather than dressed up, because the value of
    // a single phase is elsewhere: peers in a DIFFERENT phase stop
    // counting these bytes at all, which is what plan_streaming asks
    // for and what no total can express.
    session->info(fmt(
        "resource-plan: {} MB of that is phase-limited; peak across phases "
        "{} MB", phased >> 20, peak >> 20));
  }

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
        const std::string& phase, const std::string& last_phase,
        std::size_t /*floor*/) override
  {
    // No floor: an activation arena has no smaller form to fall back
    // to. It is the size the geometry makes it or the forward does not
    // run, which is why vae-decode revises the figure rather than
    // offering a range.
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
    mgr->declare_scratch(key.substr(0, bar), bytes, phase, last_phase);
    _total.fetch_add(bytes, std::memory_order_relaxed);
    _arenas.fetch_add(1, std::memory_order_relaxed);
  }

  bool
  end_plan(const SessionContextIntf* session) override
  {
    const unsigned n = _arenas.load(std::memory_order_relaxed);
    if (n == 0 || session == nullptr) { return true; }
    const std::size_t t = _total.load(std::memory_order_relaxed);
    auto* mgr = manager(session);
    const std::size_t peak = mgr != nullptr ? mgr->scratch_bytes({}) : t;
    // The COUNT, not just the bytes. An arena whose size is not knowable
    // until the first beat is declared at kUnknownArena and rounds to
    // 0 MB, and a line reading "0 MB declared" would say the opposite of
    // what happened.
    session->info(fmt(
        "resource-plan: {} activation arena(s) declared, {} MB; peak across "
        "phases {} MB", n, t >> 20, peak >> 20));
    return true;
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
