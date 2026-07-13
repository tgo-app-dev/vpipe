#include "plugin/plugin-context.h"
#include "plugin/plugin-abi.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include "vpipe-version.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
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
VpipePluginContext::register_metal_library(std::string_view name,
                                           const void*      bytes,
                                           std::size_t      n)
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  metal_compute::MetalCompute* mc =
      _session ? _session->metal_compute() : nullptr;
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
      _session ? _session->metal_compute() : nullptr;
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
