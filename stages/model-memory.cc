#include "stages/model-memory.h"

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
#include <memory>
#include <system_error>
#include <utility>

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
                 const std::vector<std::string>& dirs)
{
  namespace fs = std::filesystem;
  auto* mgr = session != nullptr ? session->services()->generative_model_manager()
                                 : nullptr;
  // Everything already open, counted once per checkpoint by the manager.
  std::size_t total = mgr != nullptr ? mgr->resident_weight_bytes() : 0;

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
               const std::string& enc_dir, std::size_t headroom)
{
  StreamPlan p;
  const std::size_t ram = phys_ram();
  if (ram == 0) { return p; }              // unknown: keep the roomy path
  p.footprint = weight_footprint(session, {dit_dir, enc_dir});
  p.others    = weight_footprint(session, {enc_dir});
  p.stream    = ram < p.footprint + headroom;
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

UnloadPolicy
parse_unload_policy(const std::string& s, bool* bad)
{
  if (bad != nullptr) { *bad = false; }
  if (s.empty() || s == "auto") { return UnloadPolicy::kAuto; }
  if (s == "always") { return UnloadPolicy::kAlways; }
  if (s == "never") { return UnloadPolicy::kNever; }
  if (bad != nullptr) { *bad = true; }
  return UnloadPolicy::kAuto;
}

const char*
unload_policy_name(UnloadPolicy p)
{
  switch (p) {
    case UnloadPolicy::kAlways: return "always";
    case UnloadPolicy::kNever:  return "never";
    default:                    return "auto";
  }
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
  }

  void
  claim(const SessionContextIntf* session, const std::string& dir) override
  {
    auto* mgr = manager(session);
    if (mgr == nullptr) { return; }
    const std::size_t b = dir_weights_bytes(dir);
    if (b == 0) { return; }        // absent component: nothing to hold
    mgr->declare_weights(dir, b);
    _declared.fetch_add(b, std::memory_order_relaxed);
  }

  void
  end_plan(const SessionContextIntf* session) override
  {
    const std::size_t d = _declared.load(std::memory_order_relaxed);
    if (d > 0 && session != nullptr) {
      session->log_debug(fmt(
          "resource-plan: declared {} MB of model weights before init",
          d >> 20));
    }
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
};

}

VPIPE_REGISTER_RESOURCE_PLANNER(ModelWeightPlanner)

}  // namespace model_memory
}  // namespace vpipe
