#include "pipeline/resource-plan.h"

#include <utility>

namespace vpipe {

ResourcePlannerRegistry&
ResourcePlannerRegistry::get() noexcept
{
  // Function-local static: constructed on first use, so a planner
  // registering from another TU's static init cannot lose a race
  // against this registry's own construction.
  static ResourcePlannerRegistry reg;
  return reg;
}

void
ResourcePlannerRegistry::add(std::unique_ptr<ResourcePlanner> p)
{
  if (p == nullptr) { return; }
  std::lock_guard<std::mutex> lk(_mu);
  for (const auto& q : _planners) {
    if (q->kind() == p->kind()) { return; }   // first registration sticks
  }
  _planners.push_back(std::move(p));
}

ResourcePlanner*
ResourcePlannerRegistry::find(std::string_view kind) const noexcept
{
  std::lock_guard<std::mutex> lk(_mu);
  for (const auto& p : _planners) {
    if (p->kind() == kind) { return p.get(); }
  }
  return nullptr;
}

std::vector<ResourcePlanner*>
ResourcePlannerRegistry::all() const
{
  std::lock_guard<std::mutex> lk(_mu);
  std::vector<ResourcePlanner*> out;
  out.reserve(_planners.size());
  for (const auto& p : _planners) { out.push_back(p.get()); }
  return out;
}

}
