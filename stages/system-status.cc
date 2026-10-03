#include "stages/system-status.h"

#include "common/soc-activity.h"
#include "common/soc-energy-channel.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

#include <dlfcn.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// ---------------------------------------------------------------------
// IOReport (private framework). On modern macOS this framework lives
// only in the dyld shared cache -- the .framework directory under
// /System/Library/PrivateFrameworks no longer exists on disk -- so we
// link it lazily via dlopen / dlsym instead of `-framework IOReport`.
// macmon and asitop use the same set of symbols.
// ---------------------------------------------------------------------
// Internal linkage: the library's GPU telemetry (gpu-telemetry.mm) keeps
// its own resolver under the same name, with a different symbol set.
namespace {
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
using MergeChannels_t = void (*)(
    CFMutableDictionaryRef a, CFDictionaryRef b, CFTypeRef c);
using ChannelGetFormat_t     = int (*)(CFDictionaryRef);
using StateGetCount_t        = int (*)(CFDictionaryRef);
using StateGetNameForIndex_t = CFStringRef (*)(CFDictionaryRef, int);
using StateGetResidency_t    = std::int64_t (*)(CFDictionaryRef, int);

// IOReportChannelGetFormat's value for a state (residency) channel.
constexpr int kFormatState = 2;

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
  // The power-manager (PMP) state channels -- see common/soc-activity.h.
  // Optional: without them the meter falls back to energy alone, which
  // is what it was before.
  bool states_ok = false;
  MergeChannels_t          merge_channels            = nullptr;
  ChannelGetGroup_t        channel_get_subgroup      = nullptr;
  ChannelGetFormat_t       channel_get_format        = nullptr;
  StateGetCount_t          state_get_count           = nullptr;
  StateGetNameForIndex_t   state_get_name_for_index  = nullptr;
  StateGetResidency_t      state_get_residency       = nullptr;
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
    r.merge_channels = reinterpret_cast<MergeChannels_t>(
        S("IOReportMergeChannels"));
    r.channel_get_subgroup = reinterpret_cast<ChannelGetGroup_t>(
        S("IOReportChannelGetSubGroup"));
    r.channel_get_format = reinterpret_cast<ChannelGetFormat_t>(
        S("IOReportChannelGetFormat"));
    r.state_get_count = reinterpret_cast<StateGetCount_t>(
        S("IOReportStateGetCount"));
    r.state_get_name_for_index = reinterpret_cast<StateGetNameForIndex_t>(
        S("IOReportStateGetNameForIndex"));
    r.state_get_residency = reinterpret_cast<StateGetResidency_t>(
        S("IOReportStateGetResidency"));
    r.states_ok = r.ok && r.merge_channels && r.channel_get_subgroup
               && r.channel_get_format && r.state_get_count
               && r.state_get_name_for_index && r.state_get_residency;
    // Intentionally never dlclose: the function pointers are kept
    // for the life of the process.
    return r;
  }();
  return a;
}

}  // namespace ioreport
}  // namespace

namespace vpipe {

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
  // M5: UNVERIFIED. Apple publishes no figure and macmon's table has no
  // M5 row, so this is the M1/M2/M4 number carried forward rather than a
  // measurement. It scales the percentage, never the watts -- the watts
  // are read from the counter -- so a wrong ceiling misreports how busy
  // the ANE is and nothing else. Stated here rather than left implicit
  // because the fallback and a real M5 entry are indistinguishable once
  // the branch exists.
  if (brand.find("M5") != std::string::npos) { return 8.0; }
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

  // PER ANE UNIT, not per box. A part with two ANEs (the `0` suffix is
  // a tile index -- see common/soc-energy-channel.h) can draw twice
  // this, and the ceiling the percentage divides by is this figure
  // times the number of ANE channels the sample actually carried.
  double                                      ane_max_w = 8.0;

  // Whether the ANE's OWN energy channels are live -- see SocEnergyGate.
  // Per source on purpose: on macOS 27 a busy GPU keeps "GPU Energy"
  // moving while ANE0 is stale, so the group moving vouches for nothing.
  SocEnergyGate                               ane_gate;

  // Add the ANE's power-manager channels to `desired`: the fabric floor
  // its links hold ("SOC Floor") and its bandwidth histograms ("AF BW").
  // The group is "PMP" on some releases and "PMP<n>" on others (the
  // same tile-index rename as the Energy Model blocks), so every
  // spelling is asked for and whatever exists is merged in. Only these
  // two subgroups, never the whole group: PMP carries some 600 channels.
  void add_pmp_channels_(const ioreport::Api& api) {
    if (!api.states_ok) { return; }
    static const char* const kGroups[]    = {"PMP", "PMP0", "PMP1",
                                             "PMP2", "PMP3"};
    static const CFStringRef kSubgroups[] = {CFSTR("SOC Floor"),
                                             CFSTR("AF BW")};
    for (const char* g : kGroups) {
      CFStringRef group = CFStringCreateWithCString(
          nullptr, g, kCFStringEncodingUTF8);
      for (CFStringRef sg : kSubgroups) {
        CFMutableDictionaryRef c =
            api.copy_channels_in_group(group, sg, 0, 0, 0);
        if (!c) { continue; }
        const void* arr = CFDictionaryGetValue(c, CFSTR("IOReportChannels"));
        const bool any = arr && CFGetTypeID(arr) == CFArrayGetTypeID() &&
                         CFArrayGetCount(static_cast<CFArrayRef>(arr)) > 0;
        if (any && desired) {
          api.merge_channels(desired, c, nullptr);
        } else if (any) {
          desired = c;
          c = nullptr;
        }
        if (c) { CFRelease(c); }
      }
      CFRelease(group);
    }
  }

  Impl() {
    ane_max_w = detect_ane_max_watts_();
    const auto& api = ioreport::resolve();
    if (!api.ok) { return; }

    desired = api.copy_channels_in_group(
        CFSTR("Energy Model"), nullptr, 0, 0, 0);
    add_pmp_channels_(api);
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

  // What one sample said about the ANE. Every field is optional on its
  // own, because each comes from a source that may be dead on this
  // machine (see common/soc-activity.h):
  //   watts    the Energy Model power -- only when that counter is live
  //   units    how many ANE energy channels the sample carried; travels
  //            with the watts because the ceiling depends on it and both
  //            come from the same sample
  //   active   share of the interval the ANE's fabric floor sat above
  //            VMIN, from the busiest of its links (0..1)
  //   bw_gbps  the ANE's fabric bandwidth, summed over its links
  struct AneSample {
    std::optional<double> watts;
    int                   units = 0;
    std::optional<double> active;
    std::optional<double> bw_gbps;
  };

  // The residencies of one state channel, named. The names point into
  // `owned`, which the caller keeps alive for as long as it reads them.
  static std::vector<SocState>
  states_of_(const ioreport::Api& api, CFDictionaryRef ch,
             std::vector<std::string>* owned) {
    const int n = api.state_get_count(ch);
    owned->clear();
    owned->reserve(n > 0 ? (std::size_t)n : 0u);
    std::vector<SocState> out;
    for (int j = 0; j < n; ++j) {
      owned->push_back(
          cf_string_to_utf8_(api.state_get_name_for_index(ch, j)));
    }
    for (int j = 0; j < n; ++j) {
      out.push_back({(*owned)[(std::size_t)j],
                     api.state_get_residency(ch, j)});
    }
    return out;
  }

  // One sample of every subscribed channel, delta against `prev`.
  std::optional<AneSample> ane_sample() {
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
    int    ane_units    = 0;
    bool   ane_moved    = false;
    AneSample out;
    std::vector<std::string> state_names;

    for (CFIndex i = 0; i < n; ++i) {
      CFDictionaryRef ch = static_cast<CFDictionaryRef>(
          CFArrayGetValueAtIndex(arr, i));
      if (!ch) { continue; }
      const std::string group =
          cf_string_to_utf8_(api.channel_get_group(ch));
      const std::string name =
          cf_string_to_utf8_(api.channel_get_channel_name(ch));
      // The ANE's power-manager channels: its fabric floor and its
      // bandwidth. A prefix match on "ANE" is right HERE, unlike for
      // energy: the floor is a maximum over links and the bandwidth a
      // sum over distinct links, neither of which a twin channel could
      // inflate -- and the links are named per release ("ANE0 L0 RD+WR",
      // "ANE L0 RD+WR", "ANE-LNK0-AF-BW").
      if (soc_block_energy_channel(group, "PMP")) {
        if (!api.states_ok || name.rfind("ANE", 0) != 0 ||
            api.channel_get_format(ch) != ioreport::kFormatState) {
          continue;
        }
        const std::string sg =
            cf_string_to_utf8_(api.channel_get_subgroup(ch));
        const std::vector<SocState> st = states_of_(api, ch, &state_names);
        if (sg == "SOC Floor") {
          if (auto a = soc_floor_active_fraction(st)) {
            out.active = std::max(out.active.value_or(0.0), *a);
          }
        } else if (sg == "AF BW" && name.size() >= 6 &&
                   name.compare(name.size() - 6, 6, " RD+WR") == 0) {
          if (auto b = soc_bw_histogram_mean_gbps(st)) {
            out.bw_gbps = out.bw_gbps.value_or(0.0) + *b;
          }
        }
        continue;
      }
      if (group != "Energy Model") { continue; }
      // "ANE" (macOS 26 and earlier) or "ANE<n>" (macOS 27 indexes every
      // block), summed across units. NOT a prefix match: that would also
      // take any future "ANE SRAM"-style companion into the total, the
      // way the GPU reader used to double-count "GPU Energy".
      if (!soc_block_energy_channel(name, "ANE")) { continue; }
      const std::int64_t raw =
          api.simple_get_integer_value(ch, 0);
      if (raw != 0) { ane_moved = true; }
      const std::string unit =
          cf_string_to_utf8_(api.channel_get_unit_label(ch));
      // Reported energy unit -> joules.
      double scale = 1.0;
      if (unit.find("mJ") != std::string::npos)      { scale = 1e-3; }
      else if (unit.find("uJ") != std::string::npos) { scale = 1e-6; }
      else if (unit.find("nJ") != std::string::npos) { scale = 1e-9; }
      else if (unit.find("J")  != std::string::npos) { scale = 1.0;  }
      ane_energy_J += static_cast<double>(raw) * scale;
      ++ane_units;
    }
    CFRelease(delta);
    // Power only from a channel known to be live: a stale one's zeros are
    // silence, not an idle ANE, and its one move in minutes is a lump.
    // The ceiling is a second, independent guard -- nothing a real ANE
    // draws comes near several times its rated power, and a lump always
    // does.
    const SocEnergyGate::Verdict v = ane_gate.feed(ane_moved);
    if (ane_units > 0 && v == SocEnergyGate::Verdict::kLive) {
      const double w = ane_energy_J / elapsed_s;
      if (w <= 4.0 * ane_max_w * (double)ane_units) { out.watts = w; }
    }
    out.units = ane_units;
    if (!out.watts && !out.active && !out.bw_gbps && ane_units == 0) {
      return std::nullopt;
    }
    return out;
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

  // ANE (IOReport). Two readings, because on any one machine one of them
  // may be dead (see common/soc-activity.h): power over a ceiling where
  // the energy counter is live, and the share of time its fabric floor
  // sat above idle where the power manager reports it. The meter is the
  // larger of the two -- a dead source reads nothing rather than a low
  // number, so the larger IS the live one, with no per-chip table.
  //
  // The ceiling is per-unit times the units this box turned out to have,
  // so a two-ANE part does not read 200% busy. Before the first sample
  // there is nothing to count, and one unit is the honest assumption --
  // every part shipped so far has one.
  auto clamp_pct = [](double p) {
    return p < 0.0 ? 0.0 : (p > 100.0 ? 100.0 : p);
  };
  const auto s = _impl->ane_sample();
  const int  units = (s && s->units > 0) ? s->units : 1;
  const double ceiling = _impl->ane_max_w * (double)units;
  oo.insert("ane_max_w", FlexData::make_real(ceiling));
  if (s) {
    std::optional<double> util;
    if (s->units > 0) {
      oo.insert("ane_units", FlexData::make_int(s->units));
    }
    if (s->watts) {
      oo.insert("ane_power_w", FlexData::make_real(*s->watts));
      util = clamp_pct(ceiling > 0.0 ? *s->watts / ceiling * 100.0 : 0.0);
    }
    if (s->active) {
      const double a = clamp_pct(*s->active * 100.0);
      oo.insert("ane_active_pct", FlexData::make_real(a));
      util = std::max(util.value_or(0.0), a);
    }
    if (s->bw_gbps) {
      oo.insert("ane_bw_gbps", FlexData::make_real(*s->bw_gbps));
    }
    if (util) { oo.insert("ane_util_pct", FlexData::make_real(*util)); }
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
