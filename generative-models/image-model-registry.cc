#include "generative-models/image-model-registry.h"

#include "pipeline/stage-config.h"
#include "pipeline/stage-consumer-types.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <exception>
#include <utility>

namespace vpipe::genai {

ImageModelRegistry&
ImageModelRegistry::get() noexcept
{
  static ImageModelRegistry instance;
  return instance;
}

bool
ImageModelRegistry::add(std::unique_ptr<ImageModelFamily> f)
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
  // A family registered here is one generate-image can now run, which is
  // a fact its static suggest_db_type cannot carry: that was written
  // when this tree was compiled and a plugin's family did not exist.
  // Telling the channel here means a plugin gets a working model picker
  // by registering a family, with nothing else to remember -- the same
  // channel the video families join, because a model picker offering
  // "diffusion models" should not care which stage will run one.
  register_channel_types("diffusion-model", tag);
  // ...and on the picker of the stage that RUNS it, whose own list is
  // a channel consumer's and does not see the channel's additions.
  register_consumer_types("generate-image", tag);
  return true;
}

ImageModelFamily*
ImageModelRegistry::find(std::string_view tag) const noexcept
{
  std::lock_guard<std::mutex> lk(_mu);
  for (const auto& e : _families) {
    if (e->tag() == tag) { return e.get(); }
  }
  return nullptr;
}

std::vector<ImageModelFamily*>
ImageModelRegistry::all() const
{
  std::lock_guard<std::mutex> lk(_mu);
  std::vector<ImageModelFamily*> out;
  out.reserve(_families.size());
  for (const auto& e : _families) { out.push_back(e.get()); }
  return out;
}

ImageModelFamily*
ImageModelRegistry::claim_for(const SessionContextIntf* session,
                             const std::string&        root,
                             const std::string&        model_type) const
{
  // Snapshot under the lock, probe outside it: `claims` reads the
  // filesystem, and holding a process-wide lock across a checkpoint
  // probe would serialise two pipelines resolving different models.
  std::vector<ImageModelFamily*> fams = all();
  for (ImageModelFamily* f : fams) {
    try {
      if (f->claims(root, model_type)) { return f; }
    } catch (const std::exception& e) {
      // A throwing probe is a bug in that family, not a reason to stop
      // asking the others -- and never a reason to take the host down.
      if (session != nullptr) {
        session->warn(fmt(
            "ImageModelRegistry: family '{}' threw probing '{}': {}; "
            "skipping it", f->tag(), root, e.what()));
      }
    } catch (...) {
      if (session != nullptr) {
        session->warn(fmt(
            "ImageModelRegistry: family '{}' threw a non-standard exception "
            "probing '{}'; skipping it", f->tag(), root));
      }
    }
  }
  return nullptr;
}

}
