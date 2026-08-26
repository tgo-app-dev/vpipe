#include "generative-models/vae-model-registry.h"

#include "pipeline/stage-config.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <exception>
#include <utility>

namespace vpipe::genai {

VaeModelRegistry&
VaeModelRegistry::get() noexcept
{
  static VaeModelRegistry instance;
  return instance;
}

bool
VaeModelRegistry::add(std::unique_ptr<VaeModelFamily> f)
{
  if (!f) { return false; }
  std::string_view tag;
  try {
    tag = f->tag();
  } catch (...) {
    return false;                     // a family that cannot name itself
  }
  if (tag.empty()) { return false; }
  // The built-in family strings vae-decode dispatches on. A plugin is
  // free to be one of these architectures, but taking the NAME makes
  // every log line ambiguous about which code decoded the clip -- and
  // dispatch is pointer-guarded, so a collision would read as a
  // built-in while running plugin code.
  for (const char* built_in : {"krea2", "flux2", "mage", "wan",
                               "minimax-h3"}) {
    if (tag == built_in) { return false; }
  }
  std::lock_guard<std::mutex> lk(_mu);
  for (const auto& e : _families) {
    if (e->tag() == tag) { return false; }   // first-wins
  }
  _families.push_back(std::move(f));
  // Same reason as the video registry: a VAE family registered here is
  // one vae-decode can now decode, and the channel is what carries that
  // to the picker. Redundant when the plugin also registered a video
  // family under the same tag -- the channel dedups -- and load-bearing
  // for an image-only family that registers just this one.
  register_channel_types("diffusion-model", tag);
  return true;
}

VaeModelFamily*
VaeModelRegistry::find(std::string_view tag) const noexcept
{
  std::lock_guard<std::mutex> lk(_mu);
  for (const auto& e : _families) {
    if (e->tag() == tag) { return e.get(); }
  }
  return nullptr;
}

std::vector<VaeModelFamily*>
VaeModelRegistry::all() const
{
  std::lock_guard<std::mutex> lk(_mu);
  std::vector<VaeModelFamily*> out;
  out.reserve(_families.size());
  for (const auto& e : _families) { out.push_back(e.get()); }
  return out;
}

VaeModelFamily*
VaeModelRegistry::claim_for(const SessionContextIntf* session,
                           const std::string&        root,
                           const std::string&        vae_dir,
                           const std::string&        model_type) const
{
  // Snapshot under the lock, probe outside it: `claims` reads the
  // filesystem, and holding a process-wide lock across a checkpoint
  // probe would serialise two pipelines resolving different models.
  std::vector<VaeModelFamily*> fams = all();
  for (VaeModelFamily* f : fams) {
    try {
      if (f->claims(root, vae_dir, model_type)) { return f; }
    } catch (const std::exception& e) {
      // A throwing probe is a bug in that family, not a reason to stop
      // asking the others -- and never a reason to take the host down.
      if (session != nullptr) {
        session->warn(fmt(
            "VaeModelRegistry: family '{}' threw probing '{}': {}; "
            "skipping it", f->tag(), root, e.what()));
      }
    } catch (...) {
      if (session != nullptr) {
        session->warn(fmt(
            "VaeModelRegistry: family '{}' threw a non-standard exception "
            "probing '{}'; skipping it", f->tag(), root));
      }
    }
  }
  return nullptr;
}

}
