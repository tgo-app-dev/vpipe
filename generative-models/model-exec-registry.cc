#include "generative-models/model-exec-registry.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <exception>
#include <utility>

namespace vpipe::genai {

ModelExecRegistry&
ModelExecRegistry::get() noexcept
{
  static ModelExecRegistry instance;
  return instance;
}

bool
ModelExecRegistry::register_arch(std::string arch, ModelExecFactory factory)
{
  if (arch.empty() || !factory) {
    return false;
  }
  std::lock_guard<std::mutex> lk(_mu);
  auto [it, inserted] = _map.try_emplace(std::move(arch), std::move(factory));
  (void)it;
  return inserted;                 // first-wins
}

bool
ModelExecRegistry::contains(std::string_view arch) const noexcept
{
  std::lock_guard<std::mutex> lk(_mu);
  return _map.find(std::string(arch)) != _map.end();
}

std::unique_ptr<ModelExec>
ModelExecRegistry::create(std::string_view           arch,
                          const ModelExecCreateArgs& args) const
{
  ModelExecFactory factory;
  {
    std::lock_guard<std::mutex> lk(_mu);
    auto it = _map.find(std::string(arch));
    if (it == _map.end()) {
      return nullptr;
    }
    factory = it->second;          // copy out; run the factory unlocked
  }
  try {
    return factory(args);
  } catch (const std::exception& e) {
    if (args.session != nullptr) {
      args.session->warn(fmt(
          "ModelExecRegistry: factory for arch '{}' threw: {}", arch,
          e.what()));
    }
    return nullptr;
  } catch (...) {
    if (args.session != nullptr) {
      args.session->warn(fmt(
          "ModelExecRegistry: factory for arch '{}' threw a non-standard "
          "exception", arch));
    }
    return nullptr;
  }
}

}
