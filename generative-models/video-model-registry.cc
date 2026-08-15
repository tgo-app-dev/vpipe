#include "generative-models/video-model-registry.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <exception>
#include <utility>

namespace vpipe::genai {

VideoModelRegistry&
VideoModelRegistry::get() noexcept
{
  static VideoModelRegistry instance;
  return instance;
}

bool
VideoModelRegistry::add(std::unique_ptr<VideoModelFamily> f)
{
  if (!f) { return false; }
  std::string_view tag;
  try {
    tag = f->tag();
  } catch (...) {
    return false;                     // a family that cannot name itself
  }
  if (tag.empty()) { return false; }
  std::lock_guard<std::mutex> lk(_mu);
  for (const auto& e : _families) {
    if (e->tag() == tag) { return false; }   // first-wins
  }
  _families.push_back(std::move(f));
  return true;
}

VideoModelFamily*
VideoModelRegistry::find(std::string_view tag) const noexcept
{
  std::lock_guard<std::mutex> lk(_mu);
  for (const auto& e : _families) {
    if (e->tag() == tag) { return e.get(); }
  }
  return nullptr;
}

std::vector<VideoModelFamily*>
VideoModelRegistry::all() const
{
  std::lock_guard<std::mutex> lk(_mu);
  std::vector<VideoModelFamily*> out;
  out.reserve(_families.size());
  for (const auto& e : _families) { out.push_back(e.get()); }
  return out;
}

VideoModelFamily*
VideoModelRegistry::claim_for(const SessionContextIntf* session,
                             const std::string&        root,
                             const std::string&        model_type) const
{
  // Snapshot under the lock, probe outside it: `claims` reads the
  // filesystem, and holding a process-wide lock across a checkpoint
  // probe would serialise two pipelines resolving different models.
  std::vector<VideoModelFamily*> fams = all();
  for (VideoModelFamily* f : fams) {
    try {
      if (f->claims(root, model_type)) { return f; }
    } catch (const std::exception& e) {
      // A throwing probe is a bug in that family, not a reason to stop
      // asking the others -- and never a reason to take the host down.
      if (session != nullptr) {
        session->warn(fmt(
            "VideoModelRegistry: family '{}' threw probing '{}': {}; "
            "skipping it", f->tag(), root, e.what()));
      }
    } catch (...) {
      if (session != nullptr) {
        session->warn(fmt(
            "VideoModelRegistry: family '{}' threw a non-standard exception "
            "probing '{}'; skipping it", f->tag(), root));
      }
    }
  }
  return nullptr;
}

}
