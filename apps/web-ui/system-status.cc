#include "apps/web-ui/system-status.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

#include <dlfcn.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <sys/sysctl.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>

// ---------------------------------------------------------------------
// IOReport (private framework). On modern macOS this framework lives
// only in the dyld shared cache -- the .framework directory under
// /System/Library/PrivateFrameworks no longer exists on disk -- so we
// link it lazily via dlopen / dlsym instead of `-framework IOReport`.
// macmon and asitop use the same set of symbols.
// ---------------------------------------------------------------------
namespace ioreport {

using SubscriptionRef = CFTypeRef;

using CopyChannelsInGroup_t = CFMutableDictionaryRef (*)(
    CFStringRef group, CFStringRef subgroup,
    std::uint64_t channel_id,
    std::uint64_t a, std::uint64_t b);
using CreateSubscription_t = SubscriptionRef (*)(
    void* allocator,
    CFMutableDictionaryRef desired,
    CFMutableDictionaryRef* sub_out,
    std::uint64_t channel_id,
    CFTypeRef a);
using CreateSamples_t = CFDictionaryRef (*)(
    SubscriptionRef sub,
    CFMutableDictionaryRef channels,
    CFTypeRef a);
using CreateSamplesDelta_t = CFDictionaryRef (*)(
    CFDictionaryRef prev,
    CFDictionaryRef cur,
    CFTypeRef a);
using ChannelGetGroup_t       = CFStringRef (*)(CFDictionaryRef);
using ChannelGetChannelName_t = CFStringRef (*)(CFDictionaryRef);
using ChannelGetUnitLabel_t   = CFStringRef (*)(CFDictionaryRef);
using SimpleGetIntegerValue_t =
    std::int64_t (*)(CFDictionaryRef, int idx);

struct Api {
  bool ok = false;
  CopyChannelsInGroup_t    copy_channels_in_group    = nullptr;
  CreateSubscription_t     create_subscription       = nullptr;
  CreateSamples_t          create_samples            = nullptr;
  CreateSamplesDelta_t     create_samples_delta      = nullptr;
  ChannelGetGroup_t        channel_get_group         = nullptr;
  ChannelGetChannelName_t  channel_get_channel_name  = nullptr;
  ChannelGetUnitLabel_t    channel_get_unit_label    = nullptr;
  SimpleGetIntegerValue_t  simple_get_integer_value  = nullptr;
};

const Api& resolve() {
  static Api a = [] {
    Api r;
    // Modern macOS ships IOReport as /usr/lib/libIOReport.dylib in
    // the dyld shared cache. Older releases (pre-Sonoma) kept it as
    // /System/Library/PrivateFrameworks/IOReport.framework/IOReport
    // -- try the dylib first and fall back to the framework path so
    // both layouts work without a recompile.
    void* h = dlopen("/usr/lib/libIOReport.dylib",
                     RTLD_LAZY | RTLD_LOCAL);
    if (!h) {
      h = dlopen(
        "/System/Library/PrivateFrameworks/IOReport.framework/IOReport",
        RTLD_LAZY | RTLD_LOCAL);
    }
    if (!h) { return r; }
    auto S = [&](const char* sym) {
      return dlsym(h, sym);
    };
    r.copy_channels_in_group =
        reinterpret_cast<CopyChannelsInGroup_t>(
            S("IOReportCopyChannelsInGroup"));
    r.create_subscription =
        reinterpret_cast<CreateSubscription_t>(
            S("IOReportCreateSubscription"));
    r.create_samples =
        reinterpret_cast<CreateSamples_t>(
            S("IOReportCreateSamples"));
    r.create_samples_delta =
        reinterpret_cast<CreateSamplesDelta_t>(
            S("IOReportCreateSamplesDelta"));
    r.channel_get_group =
        reinterpret_cast<ChannelGetGroup_t>(
            S("IOReportChannelGetGroup"));
    r.channel_get_channel_name =
        reinterpret_cast<ChannelGetChannelName_t>(
            S("IOReportChannelGetChannelName"));
    r.channel_get_unit_label =
        reinterpret_cast<ChannelGetUnitLabel_t>(
            S("IOReportChannelGetUnitLabel"));
    r.simple_get_integer_value =
        reinterpret_cast<SimpleGetIntegerValue_t>(
            S("IOReportSimpleGetIntegerValue"));
    r.ok = r.copy_channels_in_group && r.create_subscription
         && r.create_samples         && r.create_samples_delta
         && r.channel_get_group      && r.channel_get_channel_name
         && r.channel_get_unit_label && r.simple_get_integer_value;
    // Intentionally never dlclose: the function pointers are kept
    // for the life of the process.
    return r;
  }();
  return a;
}

}  // namespace ioreport

namespace vpipe::webui {

namespace {

// ----- Small CF helpers ----------------------------------------------

std::optional<double>
cf_number_for_key_(CFDictionaryRef dict, CFStringRef key)
{
  if (!dict || !key) { return std::nullopt; }
  const void* raw = CFDictionaryGetValue(dict, key);
  if (!raw) { return std::nullopt; }
  CFNumberRef num = static_cast<CFNumberRef>(raw);
  if (CFGetTypeID(num) != CFNumberGetTypeID()) { return std::nullopt; }
  double d = 0.0;
  if (!CFNumberGetValue(num, kCFNumberDoubleType, &d)) {
    return std::nullopt;
  }
  return d;
}

std::string
cf_string_to_utf8_(CFStringRef s)
{
  if (!s) { return {}; }
  // Try the fast path first; fall back to copying when CF doesn't
  // give us the internal buffer.
  if (const char* fast =
          CFStringGetCStringPtr(s, kCFStringEncodingUTF8)) {
    return std::string(fast);
  }
  CFIndex len = CFStringGetLength(s);
  CFIndex cap = CFStringGetMaximumSizeForEncoding(
      len, kCFStringEncodingUTF8) + 1;
  std::string out(cap, '\0');
  if (!CFStringGetCString(s, out.data(), cap, kCFStringEncodingUTF8)) {
    return {};
  }
  out.resize(std::strlen(out.c_str()));
  return out;
}

// Read a property that may be stored as a CFString (the modern
// layout, e.g. IOAccelerator's "model" = "Apple M4") or a CFData
// (older registry entries store the model as NUL-terminated bytes).
std::string
cf_prop_string_(CFDictionaryRef dict, CFStringRef key)
{
  if (!dict || !key) { return {}; }
  const void* raw = CFDictionaryGetValue(dict, key);
  if (!raw) { return {}; }
  const CFTypeID t = CFGetTypeID(raw);
  if (t == CFStringGetTypeID()) {
    return cf_string_to_utf8_(static_cast<CFStringRef>(raw));
  }
  if (t == CFDataGetTypeID()) {
    CFDataRef d = static_cast<CFDataRef>(raw);
    const UInt8* p = CFDataGetBytePtr(d);
    CFIndex n = CFDataGetLength(d);
    if (!p || n <= 0) { return {}; }
    while (n > 0 && p[n - 1] == 0) { --n; }   // strip trailing NULs
    return std::string(reinterpret_cast<const char*>(p),
                       static_cast<std::size_t>(n));
  }
  return {};
}

std::optional<std::uint64_t>
cf_uint_for_key_(CFDictionaryRef dict, CFStringRef key)
{
  if (!dict || !key) { return std::nullopt; }
  const void* raw = CFDictionaryGetValue(dict, key);
  if (!raw || CFGetTypeID(raw) != CFNumberGetTypeID()) {
    return std::nullopt;
  }
  long long v = 0;
  if (!CFNumberGetValue(static_cast<CFNumberRef>(raw),
                        kCFNumberLongLongType, &v)) {
    return std::nullopt;
  }
  return static_cast<std::uint64_t>(v < 0 ? 0 : v);
}

// ----- GPU stats via IOKit IORegistry --------------------------------

struct GpuStats {
  bool          ok           = false;
  double        util_pct     = 0.0;
  double        renderer_pct = 0.0;
  double        tiler_pct    = 0.0;
  std::uint64_t in_use_bytes = 0;
  std::uint64_t alloc_bytes  = 0;
  // Static descriptors (top-level IOAccelerator props, not inside
  // PerformanceStatistics): the GPU/chip model string and core count.
  std::string   model;
  std::uint64_t core_count   = 0;
  bool          has_cores    = false;
};

GpuStats
query_gpu_iokit_()
{
  GpuStats s;
  io_iterator_t iter = IO_OBJECT_NULL;
  CFMutableDictionaryRef matching = IOServiceMatching("IOAccelerator");
  if (!matching) { return s; }
  kern_return_t kr = IOServiceGetMatchingServices(
      kIOMainPortDefault, matching, &iter);
  if (kr != KERN_SUCCESS) { return s; }

  io_registry_entry_t svc = IO_OBJECT_NULL;
  while ((svc = IOIteratorNext(iter)) != IO_OBJECT_NULL) {
    CFMutableDictionaryRef props = nullptr;
    if (IORegistryEntryCreateCFProperties(
            svc, &props, kCFAllocatorDefault, 0) == KERN_SUCCESS
        && props) {
      const void* perf_raw =
          CFDictionaryGetValue(props, CFSTR("PerformanceStatistics"));
      if (perf_raw
          && CFGetTypeID(perf_raw) == CFDictionaryGetTypeID()) {
        CFDictionaryRef perf =
            static_cast<CFDictionaryRef>(perf_raw);
        if (auto v = cf_number_for_key_(
                perf, CFSTR("Device Utilization %"))) {
          s.util_pct = *v;
        }
        if (auto v = cf_number_for_key_(
                perf, CFSTR("Renderer Utilization %"))) {
          s.renderer_pct = *v;
        }
        if (auto v = cf_number_for_key_(
                perf, CFSTR("Tiler Utilization %"))) {
          s.tiler_pct = *v;
        }
        if (auto v = cf_number_for_key_(
                perf, CFSTR("In use system memory"))) {
          s.in_use_bytes = static_cast<std::uint64_t>(*v);
        }
        if (auto v = cf_number_for_key_(
                perf, CFSTR("Alloc system memory"))) {
          s.alloc_bytes = static_cast<std::uint64_t>(*v);
        }
        s.ok = true;
      }
      // Model + core count live at the top level of the accelerator
      // entry (e.g. "model" = "Apple M4", "gpu-core-count" = 10),
      // alongside PerformanceStatistics rather than inside it.
      if (s.model.empty()) {
        s.model = cf_prop_string_(props, CFSTR("model"));
      }
      if (!s.has_cores) {
        if (auto c = cf_uint_for_key_(props, CFSTR("gpu-core-count"))) {
          s.core_count = *c;
          s.has_cores  = true;
        }
      }
      CFRelease(props);
    }
    IOObjectRelease(svc);
    if (s.ok) { break; }
  }
  IOObjectRelease(iter);
  return s;
}

// ----- ANE / chip-specific ceiling (macmon table) --------------------

double
detect_ane_max_watts_()
{
  // sysctl machdep.cpu.brand_string returns "Apple M1", "Apple M2 Pro",
  // "Apple M3 Max", ... -- enough to pick the per-family ceiling. The
  // numbers match macmon's ANE table: 8.0 W for M1/M2/M4, 8.5 W for
  // the M3 family.
  char buf[256] = {0};
  std::size_t sz = sizeof(buf) - 1;
  if (sysctlbyname("machdep.cpu.brand_string", buf, &sz, nullptr, 0)
      != 0) {
    return 8.0;
  }
  std::string brand(buf);
  if (brand.find("M3") != std::string::npos) { return 8.5; }
  return 8.0;
}

// ----- Host counters -------------------------------------------------

std::uint64_t
phys_memory_bytes_()
{
  std::uint64_t total = 0;
  std::size_t sz = sizeof(total);
  if (sysctlbyname("hw.memsize", &total, &sz, nullptr, 0) != 0) {
    return 0;
  }
  return total;
}

// Activity Monitor's "Memory" column shows the process's physical
// footprint -- the bytes of compressed + uncompressed physical memory
// the kernel attributes specifically to this process (anonymous
// pages, dirty file-backed pages, compressed pool, less shared text
// pages that every Mach-O image gets for free). It's what Apple
// internally calls `phys_footprint` and exposes via task_vm_info.
//
// We previously surfaced `resident_size` (RSS) instead, which counts
// every page mapped into the address space including shared library
// text -- so for a small process it tracks the system's overall
// memory size rather than the process's footprint. Switching to
// phys_footprint makes the number match Activity Monitor's
// Memory > Memory column for this process.
std::uint64_t
phys_footprint_bytes_()
{
  task_vm_info_data_t info{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO,
                reinterpret_cast<task_info_t>(&info), &count)
      != KERN_SUCCESS) {
    return 0;
  }
  return static_cast<std::uint64_t>(info.phys_footprint);
}

// CPU/chip brand, e.g. "Apple M4" / "Apple M4 Pro". Used as a fallback
// GPU-model label when the accelerator registry omits "model".
std::string
cpu_brand_string_()
{
  char buf[256] = {0};
  std::size_t sz = sizeof(buf) - 1;
  if (sysctlbyname("machdep.cpu.brand_string", buf, &sz, nullptr, 0)
      != 0) {
    return {};
  }
  return std::string(buf);
}

}  // namespace

// ---------------------------------------------------------------------
// SystemStatusPoller::Impl
// ---------------------------------------------------------------------

struct SystemStatusPoller::Impl {
  std::mutex                                  mu;
  CFMutableDictionaryRef                      desired   = nullptr;
  CFMutableDictionaryRef                      sub_chans = nullptr;
  ioreport::SubscriptionRef                   sub       = nullptr;
  CFDictionaryRef                             prev      = nullptr;
  std::chrono::steady_clock::time_point       t_prev{};
  bool                                        ready     = false;

  double                                      ane_max_w = 8.0;

  Impl() {
    ane_max_w = detect_ane_max_watts_();
    const auto& api = ioreport::resolve();
    if (!api.ok) { return; }

    desired = api.copy_channels_in_group(
        CFSTR("Energy Model"), nullptr, 0, 0, 0);
    if (!desired) { return; }

    sub = api.create_subscription(
        nullptr, desired, &sub_chans, 0, nullptr);
    if (!sub || !sub_chans) {
      if (desired)   { CFRelease(desired);   desired   = nullptr; }
      if (sub_chans) { CFRelease(sub_chans); sub_chans = nullptr; }
      return;
    }
    // Seed the running delta with an initial sample so the very next
    // query() has something to subtract against.
    prev = api.create_samples(sub, sub_chans, nullptr);
    if (!prev) { return; }
    t_prev = std::chrono::steady_clock::now();
    ready  = true;
  }

  ~Impl() {
    if (prev)      { CFRelease(prev);      prev      = nullptr; }
    if (sub_chans) { CFRelease(sub_chans); sub_chans = nullptr; }
    if (sub)       { CFRelease(sub);       sub       = nullptr; }
    if (desired)   { CFRelease(desired);   desired   = nullptr; }
  }

  // Read one Energy Model sample, delta against `prev`, sum per-
  // channel ANE energy, divide by elapsed time -> instantaneous ANE
  // power in watts.
  std::optional<double> ane_power_w() {
    std::lock_guard<std::mutex> lk(mu);
    if (!ready) { return std::nullopt; }
    const auto& api = ioreport::resolve();
    CFDictionaryRef cur = api.create_samples(sub, sub_chans, nullptr);
    if (!cur) { return std::nullopt; }
    auto now = std::chrono::steady_clock::now();
    const double elapsed_s =
        std::chrono::duration<double>(now - t_prev).count();
    CFDictionaryRef delta =
        api.create_samples_delta(prev, cur, nullptr);
    CFRelease(prev);
    prev   = cur;
    t_prev = now;
    if (!delta || elapsed_s <= 0.0) {
      if (delta) { CFRelease(delta); }
      return std::nullopt;
    }

    const void* arr_raw =
        CFDictionaryGetValue(delta, CFSTR("IOReportChannels"));
    if (!arr_raw || CFGetTypeID(arr_raw) != CFArrayGetTypeID()) {
      CFRelease(delta);
      return std::nullopt;
    }
    CFArrayRef arr = static_cast<CFArrayRef>(arr_raw);
    const CFIndex n = CFArrayGetCount(arr);
    double ane_energy_J = 0.0;
    bool   got_ane      = false;

    for (CFIndex i = 0; i < n; ++i) {
      CFDictionaryRef ch = static_cast<CFDictionaryRef>(
          CFArrayGetValueAtIndex(arr, i));
      if (!ch) { continue; }
      const std::string group =
          cf_string_to_utf8_(api.channel_get_group(ch));
      if (group != "Energy Model") { continue; }
      const std::string name =
          cf_string_to_utf8_(api.channel_get_channel_name(ch));
      // macOS publishes the ANE channel as "ANE" (occasionally with a
      // suffix like "ANE0" on multi-tile parts). Match the prefix.
      if (name.rfind("ANE", 0) != 0) { continue; }
      const std::string unit =
          cf_string_to_utf8_(api.channel_get_unit_label(ch));
      const std::int64_t raw =
          api.simple_get_integer_value(ch, 0);
      // Reported energy unit -> joules.
      double scale = 1.0;
      if (unit.find("mJ") != std::string::npos)      { scale = 1e-3; }
      else if (unit.find("uJ") != std::string::npos) { scale = 1e-6; }
      else if (unit.find("nJ") != std::string::npos) { scale = 1e-9; }
      else if (unit.find("J")  != std::string::npos) { scale = 1.0;  }
      ane_energy_J += static_cast<double>(raw) * scale;
      got_ane = true;
    }
    CFRelease(delta);
    if (!got_ane) { return std::nullopt; }
    return ane_energy_J / elapsed_s;
  }
};

SystemStatusPoller::SystemStatusPoller()
  : _impl(std::make_unique<Impl>())
{}

SystemStatusPoller::~SystemStatusPoller() = default;

FlexData
SystemStatusPoller::query()
{
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();

  // GPU (IORegistry).
  const auto gpu = query_gpu_iokit_();
  if (gpu.ok) {
    oo.insert("gpu_util_pct",     FlexData::make_real(gpu.util_pct));
    oo.insert("gpu_renderer_pct", FlexData::make_real(gpu.renderer_pct));
    oo.insert("gpu_tiler_pct",    FlexData::make_real(gpu.tiler_pct));
    oo.insert("gpu_in_use_bytes",
              FlexData::make_uint(gpu.in_use_bytes));
    oo.insert("gpu_alloc_bytes",
              FlexData::make_uint(gpu.alloc_bytes));
  }

  // GPU model + core count for the status bar's machine label, e.g.
  // "Apple M4 (10)". These come from the accelerator entry even when
  // PerformanceStatistics is unavailable; fall back to the CPU brand
  // string for the model so the label is never empty on Apple Silicon.
  std::string model = gpu.model;
  if (model.empty()) { model = cpu_brand_string_(); }
  if (!model.empty()) {
    oo.insert("gpu_model", FlexData::make_string(model));
  }
  if (gpu.has_cores) {
    oo.insert("gpu_cores", FlexData::make_uint(gpu.core_count));
  }

  // ANE (IOReport).
  oo.insert("ane_max_w", FlexData::make_real(_impl->ane_max_w));
  if (auto w = _impl->ane_power_w()) {
    const double pct = (*w / _impl->ane_max_w) * 100.0;
    oo.insert("ane_power_w", FlexData::make_real(*w));
    oo.insert("ane_util_pct",
        FlexData::make_real(pct < 0.0 ? 0.0 : (pct > 100.0 ? 100.0 : pct)));
  }

  // This process's physical footprint -- the number Activity
  // Monitor labels "Memory" for the row -- plus the host total for
  // context.
  oo.insert("phys_footprint_bytes",
            FlexData::make_uint(phys_footprint_bytes_()));
  oo.insert("phys_total_bytes",
            FlexData::make_uint(phys_memory_bytes_()));

  return o;
}

}
