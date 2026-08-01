#include "ui/ui-view-registry.h"

#include <utility>

namespace vpipe {

UiViewRegistry&
UiViewRegistry::get() noexcept
{
  // Function-local static: initialized on first use, so a view's
  // static-init registration can never race the registry's own
  // construction regardless of TU initialization order.
  static UiViewRegistry reg;
  return reg;
}

void
UiViewRegistry::register_view(const UiViewSpec* spec) noexcept
{
  if (spec == nullptr || spec->id.empty()) { return; }
  std::lock_guard<std::mutex> lk(_mu);
  auto [it, inserted] = _by_id.emplace(std::string(spec->id), spec);
  (void)it;
  if (!inserted) { return; }   // first registration of an id wins
  _views.push_back(spec);
}

const UiViewSpec*
UiViewRegistry::find(std::string_view id) const noexcept
{
  std::lock_guard<std::mutex> lk(_mu);
  auto it = _by_id.find(id);
  return it == _by_id.end() ? nullptr : it->second;
}

std::vector<const UiViewSpec*>
UiViewRegistry::all() const
{
  std::lock_guard<std::mutex> lk(_mu);
  return _views;
}

void
UiViewRegistry::register_assets(const UiViewAsset* assets,
                                std::size_t        n) noexcept
{
  if (assets == nullptr) { return; }
  std::lock_guard<std::mutex> lk(_mu);
  for (std::size_t i = 0; i < n; ++i) {
    if (assets[i].path.empty()) { continue; }
    _assets.emplace(std::string(assets[i].path), &assets[i]);
  }
}

const UiViewAsset*
UiViewRegistry::find_asset(std::string_view path) const noexcept
{
  std::lock_guard<std::mutex> lk(_mu);
  auto it = _assets.find(path);
  return it == _assets.end() ? nullptr : it->second;
}

}
