#include "plugin/plugin-context.h"
#include "plugin/plugin-abi.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"

#include "vpipe-version.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
// The whole generative-models subsystem -- the video family registry with
// it -- builds under VPIPE_BUILD_APPLE_SILICON, so this include and the
// registry call below carry the same gate metal libraries do.
#include "generative-models/vae-model-registry.h"
#include "generative-models/video-model-registry.h"
#endif

namespace vpipe {

namespace {

// libvpipe version string, e.g. "0.1 (a1b2c3d4*0)". Built once from the
// generated vpipe-version.h macros (available on the build include path
// because this TU compiles into libvpipe).
const std::string&
host_version_string_()
{
  static const std::string v =
      std::string(VPIPE_VERSION_MAJOR) + "." + VPIPE_VERSION_MINOR
      + " (" + GIT_HASH + ")";
  return v;
}

}  // namespace

VpipePluginContext::VpipePluginContext(const SessionContextIntf* session,
                                       std::string_view          plugin_name)
  : _session(session), _plugin(plugin_name)
{
}

std::uint32_t
VpipePluginContext::abi_version() const noexcept
{
  return VPIPE_PLUGIN_ABI_VERSION;
}

std::string_view
VpipePluginContext::host_version() const noexcept
{
  return host_version_string_();
}

void
VpipePluginContext::register_stage(std::string_view       type_name,
                                   StageRegistry::Factory factory,
                                   const StageSpec*       spec)
{
  // register_type is first-wins and idempotent (a TypedStage also self-
  // registers at load). set_spec attaches the formal description.
  StageRegistry::get().register_type(type_name, factory);
  if (spec != nullptr) {
    StageRegistry::get().set_spec(type_name, spec);
  }
  if (_session != nullptr) {
    _session->log_normal(fmt(
        "plugin '{}': registered stage type '{}'", _plugin, type_name));
  }
}

bool
VpipePluginContext::register_video_family(
    std::unique_ptr<genai::VideoModelFamily> family)
{
  if (!family) { return false; }
#ifdef VPIPE_BUILD_APPLE_SILICON
  // Read the tag BEFORE the move: after it, `family` is null.
  const std::string tag(family->tag());
  const bool ok =
      genai::VideoModelRegistry::get().add(std::move(family));
  if (_session != nullptr) {
    if (ok) {
      _session->log_normal(fmt(
          "plugin '{}': registered video model family '{}'", _plugin, tag));
    } else {
      _session->warn(fmt(
          "plugin '{}': video model family '{}' was NOT registered -- that "
          "tag is already taken (or the family could not name itself). The "
          "family already present keeps the tag", _plugin, tag));
    }
  }
  return ok;
#else
  // `family` is destroyed here, which needs the type complete -- and on
  // a non-apple build it is not. Leak the pointer rather than pretend:
  // this branch cannot be reached by a working plugin (there is no
  // generate-video to register with), and a plugin that gets here is
  // already misbuilt.
  (void)family.release();
  if (_session != nullptr) {
    _session->warn(fmt(
        "plugin '{}': video model families are unsupported in this build",
        _plugin));
  }
  return false;
#endif
}

bool
VpipePluginContext::register_vae_family(
    std::unique_ptr<genai::VaeModelFamily> family)
{
  if (!family) { return false; }
#ifdef VPIPE_BUILD_APPLE_SILICON
  // Read the tag BEFORE the move: after it, `family` is null.
  const std::string tag(family->tag());
  const bool ok =
      genai::VaeModelRegistry::get().add(std::move(family));
  if (_session != nullptr) {
    if (ok) {
      _session->log_normal(fmt(
          "plugin '{}': registered VAE family '{}'", _plugin, tag));
    } else {
      _session->warn(fmt(
          "plugin '{}': VAE family '{}' was NOT registered -- that "
          "tag is already taken, collides with a built-in family name, (or the family could not name itself). The "
          "family already present keeps the tag", _plugin, tag));
    }
  }
  return ok;
#else
  // `family` is destroyed here, which needs the type complete -- and on
  // a non-apple build it is not. Leak the pointer rather than pretend:
  // this branch cannot be reached by a working plugin (there is no
  // vae-decode to register with), and a plugin that gets here is
  // already misbuilt.
  (void)family.release();
  if (_session != nullptr) {
    _session->warn(fmt(
        "plugin '{}': VAE families are unsupported in this build",
        _plugin));
  }
  return false;
#endif
}

std::size_t
VpipePluginContext::register_catalog_entries(
    std::vector<ModelCatalogEntry> entries)
{
  const std::size_t asked = entries.size();
  const std::size_t taken = ::vpipe::register_catalog_entries(std::move(entries));
  if (_session != nullptr) {
    if (taken == asked) {
      _session->log_normal(fmt(
          "plugin '{}': added {} model catalogue entries", _plugin, taken));
    } else {
      // Not an error -- two plugins can legitimately ship the same repo --
      // but a silent drop would leave a menu row missing with no reason.
      _session->log_normal(fmt(
          "plugin '{}': added {} of {} model catalogue entries ({} already "
          "catalogued)", _plugin, taken, asked, asked - taken));
    }
  }
  return taken;
}

bool
VpipePluginContext::register_metal_library(std::string_view name,
                                           const void*      bytes,
                                           std::size_t      n)
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  metal_compute::MetalCompute* mc =
      _session ? _session->services()->metal_compute() : nullptr;
  if (mc == nullptr || !mc->valid()) {
    if (_session != nullptr) {
      _session->warn(fmt(
          "plugin '{}': metal-compute unavailable; cannot register metal "
          "library '{}'", _plugin, name));
    }
    return false;
  }
  return mc->register_metal_library(
      name, static_cast<const unsigned char*>(bytes), n);
#else
  (void)bytes;
  (void)n;
  if (_session != nullptr) {
    _session->warn(fmt(
        "plugin '{}': metal libraries are unsupported in this build "
        "(cannot register '{}')", _plugin, name));
  }
  return false;
#endif
}

bool
VpipePluginContext::register_metal_library_file(std::string_view name,
                                                std::string_view path)
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  metal_compute::MetalCompute* mc =
      _session ? _session->services()->metal_compute() : nullptr;
  if (mc == nullptr || !mc->valid()) {
    if (_session != nullptr) {
      _session->warn(fmt(
          "plugin '{}': metal-compute unavailable; cannot register metal "
          "library '{}'", _plugin, name));
    }
    return false;
  }
  return mc->register_metal_library_file(name, path);
#else
  (void)path;
  if (_session != nullptr) {
    _session->warn(fmt(
        "plugin '{}': metal libraries are unsupported in this build "
        "(cannot register '{}')", _plugin, name));
  }
  return false;
#endif
}

}
