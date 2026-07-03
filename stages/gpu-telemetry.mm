#include "stages/gpu-telemetry.h"

#import <Foundation/Foundation.h>

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

#include <dlfcn.h>
#include <mach/mach.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------
// IOReport (private framework). Resolved lazily via dlopen/dlsym -- on
// modern macOS it lives only in the dyld shared cache as
// /usr/lib/libIOReport.dylib (no on-disk .framework). Same symbol set
// macmon / asitop consume. Beyond the "simple" report accessors that
// system-status.cc uses for ANE energy, the GPU pstate channel is a
// STATE report, so we also bind the state-report accessors.
// ---------------------------------------------------------------------
namespace ioreport {

using SubscriptionRef = CFTypeRef;

using CopyChannelsInGroup_t = CFMutableDictionaryRef (*)(
    CFStringRef group, CFStringRef subgroup,
    std::uint64_t channel_id, std::uint64_t a, std::uint64_t b);
using CreateSubscription_t = SubscriptionRef (*)(
    void* allocator, CFMutableDictionaryRef desired,
    CFMutableDictionaryRef* sub_out, std::uint64_t channel_id,
    CFTypeRef a);
using CreateSamples_t = CFDictionaryRef (*)(
    SubscriptionRef sub, CFMutableDictionaryRef channels, CFTypeRef a);
using CreateSamplesDelta_t = CFDictionaryRef (*)(
    CFDictionaryRef prev, CFDictionaryRef cur, CFTypeRef a);
using ChannelGetGroup_t       = CFStringRef (*)(CFDictionaryRef);
using ChannelGetChannelName_t = CFStringRef (*)(CFDictionaryRef);
using ChannelGetUnitLabel_t   = CFStringRef (*)(CFDictionaryRef);
using SimpleGetIntegerValue_t =
    std::int64_t (*)(CFDictionaryRef, int idx);
using StateGetCount_t        = int (*)(CFDictionaryRef);
using StateGetNameForIndex_t = CFStringRef (*)(CFDictionaryRef, int);
using StateGetResidency_t    = std::int64_t (*)(CFDictionaryRef, int);

struct Api {
  bool ok = false;
  CopyChannelsInGroup_t    copy_channels_in_group   = nullptr;
  CreateSubscription_t     create_subscription      = nullptr;
  CreateSamples_t          create_samples           = nullptr;
  CreateSamplesDelta_t     create_samples_delta     = nullptr;
  ChannelGetGroup_t        channel_get_group        = nullptr;
  ChannelGetChannelName_t  channel_get_channel_name = nullptr;
  ChannelGetUnitLabel_t    channel_get_unit_label   = nullptr;
  SimpleGetIntegerValue_t  simple_get_integer_value = nullptr;
  StateGetCount_t          state_get_count          = nullptr;
  StateGetNameForIndex_t   state_get_name_for_index = nullptr;
  StateGetResidency_t      state_get_residency      = nullptr;
};

const Api& resolve() {
  static Api a = [] {
    Api r;
    void* h = dlopen("/usr/lib/libIOReport.dylib",
                     RTLD_LAZY | RTLD_LOCAL);
    if (!h) {
      h = dlopen(
        "/System/Library/PrivateFrameworks/IOReport.framework/IOReport",
        RTLD_LAZY | RTLD_LOCAL);
    }
    if (!h) { return r; }
    auto S = [&](const char* sym) { return dlsym(h, sym); };
    r.copy_channels_in_group =
        reinterpret_cast<CopyChannelsInGroup_t>(
            S("IOReportCopyChannelsInGroup"));
    r.create_subscription =
        reinterpret_cast<CreateSubscription_t>(
            S("IOReportCreateSubscription"));
    r.create_samples =
        reinterpret_cast<CreateSamples_t>(S("IOReportCreateSamples"));
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
    r.state_get_count =
        reinterpret_cast<StateGetCount_t>(
            S("IOReportStateGetCount"));
    r.state_get_name_for_index =
        reinterpret_cast<StateGetNameForIndex_t>(
            S("IOReportStateGetNameForIndex"));
    r.state_get_residency =
        reinterpret_cast<StateGetResidency_t>(
            S("IOReportStateGetResidency"));
    r.ok = r.copy_channels_in_group && r.create_subscription
        && r.create_samples         && r.create_samples_delta
        && r.channel_get_group      && r.channel_get_channel_name
        && r.channel_get_unit_label && r.simple_get_integer_value;
    return r;
  }();
  return a;
}

}  // namespace ioreport

namespace vpipe {

namespace {

// ----- Small CF helpers (mirrors system-status.cc) -------------------

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
  if (const char* fast =
          CFStringGetCStringPtr(s, kCFStringEncodingUTF8)) {
    return std::string(fast);
  }
  CFIndex len = CFStringGetLength(s);
  CFIndex cap = CFStringGetMaximumSizeForEncoding(
      len, kCFStringEncodingUTF8) + 1;
  std::string out(cap, '\0');
  if (!CFStringGetCString(s, out.data(), cap,
                          kCFStringEncodingUTF8)) {
    return {};
  }
  out.resize(std::strlen(out.c_str()));
  return out;
}

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
    while (n > 0 && p[n - 1] == 0) { --n; }
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

// ----- GPU util / model / core count via IOAccelerator ---------------

struct GpuInfo {
  bool          util_ok    = false;
  double        util_pct   = 0.0;
  std::string   model;
  std::uint64_t core_count = 0;
  bool          has_cores  = false;
};

GpuInfo
query_gpu_iokit_()
{
  GpuInfo s;
  CFMutableDictionaryRef matching =
      IOServiceMatching("IOAccelerator");
  if (!matching) { return s; }
  io_iterator_t iter = IO_OBJECT_NULL;
  if (IOServiceGetMatchingServices(
          kIOMainPortDefault, matching, &iter) != KERN_SUCCESS) {
    return s;
  }
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
          s.util_ok  = true;
        }
      }
      if (s.model.empty()) {
        s.model = cf_prop_string_(props, CFSTR("model"));
      }
      if (!s.has_cores) {
        if (auto c =
                cf_uint_for_key_(props, CFSTR("gpu-core-count"))) {
          s.core_count = *c;
          s.has_cores  = true;
        }
      }
      CFRelease(props);
    }
    IOObjectRelease(svc);
    if (s.util_ok && !s.model.empty()) { break; }
  }
  IOObjectRelease(iter);
  return s;
}

// ----- GPU DVFS MHz table from the pmgr IORegistry node --------------
// The GPU performance-state frequency table is published as
// "voltage-states9": a CFData of (u32 freq_hz, u32 voltage) LE pairs on
// the AppleARMIODevice node named "pmgr". Entry 0 is the idle state
// (freq 0). Returns MHz per pstate, index-aligned with the GPUPH state
// residency report; empty if not resolvable on this SoC.
std::vector<double>
read_gpu_mhz_table_()
{
  std::vector<double> table;
  CFMutableDictionaryRef matching =
      IOServiceMatching("AppleARMIODevice");
  if (!matching) { return table; }
  io_iterator_t iter = IO_OBJECT_NULL;
  if (IOServiceGetMatchingServices(
          kIOMainPortDefault, matching, &iter) != KERN_SUCCESS) {
    return table;
  }
  io_registry_entry_t svc = IO_OBJECT_NULL;
  while ((svc = IOIteratorNext(iter)) != IO_OBJECT_NULL) {
    CFMutableDictionaryRef props = nullptr;
    if (IORegistryEntryCreateCFProperties(
            svc, &props, kCFAllocatorDefault, 0) == KERN_SUCCESS
        && props) {
      const std::string nm = cf_prop_string_(props, CFSTR("name"));
      if (nm == "pmgr") {
        const void* raw =
            CFDictionaryGetValue(props, CFSTR("voltage-states9"));
        if (raw && CFGetTypeID(raw) == CFDataGetTypeID()) {
          CFDataRef d = static_cast<CFDataRef>(raw);
          const UInt8* p = CFDataGetBytePtr(d);
          const CFIndex n = CFDataGetLength(d);
          for (CFIndex off = 0; p && off + 8 <= n; off += 8) {
            std::uint32_t f = 0;
            std::memcpy(&f, p + off, 4);   // LE freq, Hz
            table.push_back(static_cast<double>(f) / 1e6);
          }
        }
      }
      CFRelease(props);
    }
    IOObjectRelease(svc);
    if (!table.empty()) { break; }
  }
  IOObjectRelease(iter);
  return table;
}

// ----- AppleSMC GPU die temperature ----------------------------------
// The classic IOConnectCallStructMethod dance against AppleSMC.

constexpr std::uint32_t kSmcIndex        = 2;   // kSMCHandleYPCEvent
constexpr std::uint8_t  kSmcCmdReadBytes = 5;
constexpr std::uint8_t  kSmcCmdReadIndex = 8;
constexpr std::uint8_t  kSmcCmdKeyInfo   = 9;

// NOTE: these structs are laid out with NATURAL alignment to match the
// AppleSMC kernel ABI -- do NOT #pragma pack them (packing them silently
// breaks every IOConnectCallStructMethod, which then returns zeroed
// output).
struct SmcVers {
  std::uint8_t  major, minor, build, reserved;
  std::uint16_t release;
};
struct SmcPLimit {
  std::uint16_t version;
  std::uint16_t length;
  std::uint32_t cpu_plimit, gpu_plimit, mem_plimit;
};
struct SmcKeyInfo {
  std::uint32_t data_size;
  std::uint32_t data_type;
  std::uint8_t  data_attributes;
};
struct SmcKeyData {
  std::uint32_t key;
  SmcVers       vers;
  SmcPLimit     plimit;
  SmcKeyInfo    key_info;
  std::uint8_t  result;
  std::uint8_t  status;
  std::uint8_t  data8;
  std::uint32_t data32;
  std::uint8_t  bytes[32];
};

std::uint32_t
fourcc_(const char* s)
{
  return (static_cast<std::uint32_t>(s[0]) << 24) |
         (static_cast<std::uint32_t>(s[1]) << 16) |
         (static_cast<std::uint32_t>(s[2]) << 8) |
         static_cast<std::uint32_t>(s[3]);
}

class Smc {
public:
  Smc()
  {
    io_service_t svc = IOServiceGetMatchingService(
        kIOMainPortDefault, IOServiceMatching("AppleSMC"));
    if (svc == IO_OBJECT_NULL) { return; }
    if (IOServiceOpen(svc, mach_task_self(), 0, &_conn)
        != KERN_SUCCESS) {
      _conn = IO_OBJECT_NULL;
    }
    IOObjectRelease(svc);
    if (_conn != IO_OBJECT_NULL) { discover_keys_(); }
  }
  ~Smc()
  {
    if (_conn != IO_OBJECT_NULL) { IOServiceClose(_conn); }
  }
  bool ok() const { return _conn != IO_OBJECT_NULL && !_keys.empty(); }

  // Average GPU/SoC die temperature (C) over the discovered sensors, or
  // nullopt if none read a plausible value.
  std::optional<double> read_gpu_temp()
  {
    if (!ok()) { return std::nullopt; }
    double sum = 0.0;
    int    n   = 0;
    for (std::uint32_t k : _keys) {
      if (auto v = read_temp_key_(k)) { sum += *v; ++n; }
    }
    if (n == 0) { return std::nullopt; }
    return sum / static_cast<double>(n);
  }

private:
  bool call_(const SmcKeyData& in, SmcKeyData& out)
  {
    std::size_t out_sz = sizeof(out);
    return IOConnectCallStructMethod(
               _conn, kSmcIndex, &in, sizeof(in), &out, &out_sz)
           == KERN_SUCCESS;
  }

  // Enumerate the SMC key table and keep the temperature sensors that
  // read a plausible value. The documented GPU prefix "Tg" wins when it
  // yields real readings; on SoCs where it doesn't (e.g. M4 Pro reports
  // -4/0/2.4 there) we fall back to the "Tp" SoC die cluster (the
  // sensors that actually track GPU/SoC heat).
  void discover_keys_()
  {
    const std::uint32_t count = key_count_();
    std::vector<std::uint32_t> tg, tp;
    for (std::uint32_t i = 0; i < count; ++i) {
      const std::uint32_t key = key_at_index_(i);
      if (!key) { continue; }
      char s[5];
      key_to_str_(key, s);
      const bool is_tg = s[0] == 'T' && (s[1] == 'g' || s[1] == 'G');
      const bool is_tp = s[0] == 'T' &&  s[1] == 'p';
      if (!is_tg && !is_tp) { continue; }
      if (!read_temp_key_(key)) { continue; }   // implausible -> drop
      (is_tg ? tg : tp).push_back(key);
    }
    _keys = !tg.empty() ? std::move(tg) : std::move(tp);
  }

  std::uint32_t key_count_()
  {
    SmcKeyData in{};
    SmcKeyData out{};
    in.key   = fourcc_("#KEY");
    in.data8 = kSmcCmdKeyInfo;
    if (!call_(in, out)) { return 0; }
    SmcKeyData rd{};
    SmcKeyData ro{};
    rd.key                = fourcc_("#KEY");
    rd.key_info.data_size = out.key_info.data_size;
    rd.data8              = kSmcCmdReadBytes;
    if (!call_(rd, ro)) { return 0; }
    return (static_cast<std::uint32_t>(ro.bytes[0]) << 24) |
           (static_cast<std::uint32_t>(ro.bytes[1]) << 16) |
           (static_cast<std::uint32_t>(ro.bytes[2]) << 8) |
           static_cast<std::uint32_t>(ro.bytes[3]);
  }

  std::uint32_t key_at_index_(std::uint32_t idx)
  {
    SmcKeyData in{};
    SmcKeyData out{};
    in.data8  = kSmcCmdReadIndex;
    in.data32 = idx;
    if (!call_(in, out)) { return 0; }
    return out.key;
  }

  // Read one key as a temperature; nullopt if absent / implausible.
  std::optional<double> read_temp_key_(std::uint32_t key)
  {
    SmcKeyData in{};
    SmcKeyData out{};
    in.key   = key;
    in.data8 = kSmcCmdKeyInfo;
    if (!call_(in, out)) { return std::nullopt; }
    const std::uint32_t size = out.key_info.data_size;
    const std::uint32_t type = out.key_info.data_type;
    if (size == 0 || size > sizeof(out.bytes)) {
      return std::nullopt;
    }
    SmcKeyData rd{};
    SmcKeyData rout{};
    rd.key                = key;
    rd.key_info.data_size = size;
    rd.data8              = kSmcCmdReadBytes;
    if (!call_(rd, rout)) { return std::nullopt; }
    const double v = decode_(rout.bytes, size, type);
    if (v < 20.0 || v > 110.0) { return std::nullopt; }
    return v;
  }

  static void key_to_str_(std::uint32_t k, char* o)
  {
    o[0] = static_cast<char>(k >> 24);
    o[1] = static_cast<char>(k >> 16);
    o[2] = static_cast<char>(k >> 8);
    o[3] = static_cast<char>(k);
    o[4] = '\0';
  }

  // SMC numeric encodings: "flt " IEEE float (LE), "ioft" 64-bit
  // fixed-point (16 frac bits, BE), "sp78" signed 8.8 (BE).
  static double decode_(const std::uint8_t* b, std::uint32_t size,
                        std::uint32_t type)
  {
    if (type == fourcc_("flt ") && size == 4) {
      float f = 0.0f;
      std::memcpy(&f, b, 4);
      return static_cast<double>(f);
    }
    if (type == fourcc_("ioft") && size == 8) {
      std::uint64_t u = 0;
      for (std::uint32_t i = 0; i < 8; ++i) {
        u = (u << 8) | b[i];
      }
      return static_cast<double>(u) / 65536.0;
    }
    if (type == fourcc_("sp78") && size == 2) {
      std::int16_t s = static_cast<std::int16_t>((b[0] << 8) | b[1]);
      return static_cast<double>(s) / 256.0;
    }
    return -1.0;
  }

  io_connect_t               _conn = IO_OBJECT_NULL;
  std::vector<std::uint32_t> _keys;
};

// ----- OS thermal-pressure label -------------------------------------

std::string
thermal_state_label_()
{
  NSProcessInfoThermalState s =
      [[NSProcessInfo processInfo] thermalState];
  switch (s) {
    case NSProcessInfoThermalStateNominal:  return "Nominal";
    case NSProcessInfoThermalStateFair:     return "Fair";
    case NSProcessInfoThermalStateSerious:  return "Serious";
    case NSProcessInfoThermalStateCritical: return "Critical";
  }
  return "Unknown";
}

// ----- Process memory footprint --------------------------------------

// Whole-process physical footprint (what Activity Monitor's "Memory"
// column shows), in MB. On the UMA this includes the wired model
// weights, the KV cache and activations, so it tracks the model's
// resident size and its growth with context.
std::optional<double>
process_footprint_mb_()
{
  task_vm_info_data_t info{};
  mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO,
                reinterpret_cast<task_info_t>(&info), &cnt)
      != KERN_SUCCESS) {
    return std::nullopt;
  }
  return static_cast<double>(info.phys_footprint)
         / (1024.0 * 1024.0);
}

// ----- Streaming aggregate -------------------------------------------

struct Acc {
  int    n   = 0;
  double sum = 0.0;
  double mn  = 0.0;
  double mx  = 0.0;
  bool   any = false;
  void add(double v)
  {
    if (!any) { mn = mx = v; any = true; }
    else      { mn = std::min(mn, v); mx = std::max(mx, v); }
    sum += v;
    ++n;
  }
  MetricAgg done() const
  {
    MetricAgg a;
    a.ok = any;
    a.n  = n;
    if (any) {
      a.min = mn;
      a.max = mx;
      a.avg = sum / static_cast<double>(n);
    }
    return a;
  }
};

}  // namespace

// ---------------------------------------------------------------------
// GpuTelemetrySampler::Impl
// ---------------------------------------------------------------------

struct GpuTelemetrySampler::Impl {
  // IOReport energy (power) subscription.
  CFMutableDictionaryRef pwr_desired = nullptr;
  CFMutableDictionaryRef pwr_chans   = nullptr;
  ioreport::SubscriptionRef pwr_sub  = nullptr;
  CFDictionaryRef        pwr_prev    = nullptr;
  std::chrono::steady_clock::time_point pwr_t{};
  bool pwr_ready = false;

  // IOReport GPU-stats (frequency) subscription.
  CFMutableDictionaryRef frq_desired = nullptr;
  CFMutableDictionaryRef frq_chans   = nullptr;
  ioreport::SubscriptionRef frq_sub  = nullptr;
  CFDictionaryRef        frq_prev    = nullptr;
  std::vector<double>    mhz_table;
  bool frq_ready = false;

  Smc smc;

  std::thread        worker;
  std::atomic<bool>  running{false};

  Acc         a_freq, a_temp, a_util, a_pwr, a_mem;
  std::string thermal;

  Impl()
  {
    mhz_table = read_gpu_mhz_table_();
    const auto& api = ioreport::resolve();
    if (api.ok) {
      pwr_desired = api.copy_channels_in_group(
          CFSTR("Energy Model"), nullptr, 0, 0, 0);
      if (pwr_desired) {
        pwr_sub = api.create_subscription(
            nullptr, pwr_desired, &pwr_chans, 0, nullptr);
        if (pwr_sub && pwr_chans) { pwr_ready = true; }
      }
      frq_desired = api.copy_channels_in_group(
          CFSTR("GPU Stats"), nullptr, 0, 0, 0);
      if (frq_desired) {
        frq_sub = api.create_subscription(
            nullptr, frq_desired, &frq_chans, 0, nullptr);
        if (frq_sub && frq_chans && api.state_get_count
            && api.state_get_residency && !mhz_table.empty()) {
          frq_ready = true;
        }
      }
    }
  }

  ~Impl()
  {
    if (running.load()) { join_(); }
    if (pwr_prev)    { CFRelease(pwr_prev); }
    if (pwr_chans)   { CFRelease(pwr_chans); }
    if (pwr_sub)     { CFRelease(pwr_sub); }
    if (pwr_desired) { CFRelease(pwr_desired); }
    if (frq_prev)    { CFRelease(frq_prev); }
    if (frq_chans)   { CFRelease(frq_chans); }
    if (frq_sub)     { CFRelease(frq_sub); }
    if (frq_desired) { CFRelease(frq_desired); }
  }

  // One GPU energy delta -> watts.
  std::optional<double> sample_power_()
  {
    if (!pwr_ready) { return std::nullopt; }
    const auto& api = ioreport::resolve();
    CFDictionaryRef cur = api.create_samples(pwr_sub, pwr_chans, nullptr);
    if (!cur) { return std::nullopt; }
    auto now = std::chrono::steady_clock::now();
    const double elapsed_s =
        std::chrono::duration<double>(now - pwr_t).count();
    CFDictionaryRef delta =
        api.create_samples_delta(pwr_prev, cur, nullptr);
    CFRelease(pwr_prev);
    pwr_prev = cur;
    pwr_t    = now;
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
    double energy_J = 0.0;
    bool   got      = false;
    for (CFIndex i = 0; i < n; ++i) {
      CFDictionaryRef ch = static_cast<CFDictionaryRef>(
          CFArrayGetValueAtIndex(arr, i));
      if (!ch) { continue; }
      if (cf_string_to_utf8_(api.channel_get_group(ch))
          != "Energy Model") { continue; }
      const std::string name =
          cf_string_to_utf8_(api.channel_get_channel_name(ch));
      if (name.rfind("GPU", 0) != 0) { continue; }
      const std::string unit =
          cf_string_to_utf8_(api.channel_get_unit_label(ch));
      const std::int64_t raw = api.simple_get_integer_value(ch, 0);
      double scale = 1.0;
      if      (unit.find("mJ") != std::string::npos) { scale = 1e-3; }
      else if (unit.find("uJ") != std::string::npos) { scale = 1e-6; }
      else if (unit.find("nJ") != std::string::npos) { scale = 1e-9; }
      else if (unit.find("J")  != std::string::npos) { scale = 1.0;  }
      energy_J += static_cast<double>(raw) * scale;
      got = true;
    }
    CFRelease(delta);
    if (!got) { return std::nullopt; }
    return energy_J / elapsed_s;
  }

  // Residency-weighted GPU active frequency (MHz) over the interval.
  std::optional<double> sample_freq_()
  {
    if (!frq_ready) { return std::nullopt; }
    const auto& api = ioreport::resolve();
    CFDictionaryRef cur = api.create_samples(frq_sub, frq_chans, nullptr);
    if (!cur) { return std::nullopt; }
    CFDictionaryRef delta =
        api.create_samples_delta(frq_prev, cur, nullptr);
    CFRelease(frq_prev);
    frq_prev = cur;
    if (!delta) { return std::nullopt; }
    const void* arr_raw =
        CFDictionaryGetValue(delta, CFSTR("IOReportChannels"));
    if (!arr_raw || CFGetTypeID(arr_raw) != CFArrayGetTypeID()) {
      CFRelease(delta);
      return std::nullopt;
    }
    CFArrayRef arr = static_cast<CFArrayRef>(arr_raw);
    const CFIndex n = CFArrayGetCount(arr);
    std::optional<double> result;
    for (CFIndex i = 0; i < n && !result; ++i) {
      CFDictionaryRef ch = static_cast<CFDictionaryRef>(
          CFArrayGetValueAtIndex(arr, i));
      if (!ch) { continue; }
      if (cf_string_to_utf8_(api.channel_get_group(ch))
          != "GPU Stats") { continue; }
      const std::string name =
          cf_string_to_utf8_(api.channel_get_channel_name(ch));
      // The pstate residency histogram (commonly "GPUPH").
      if (name.find("PH") == std::string::npos) { continue; }
      const int count = api.state_get_count(ch);
      if (count <= 1) { continue; }
      double wsum = 0.0;
      double rsum = 0.0;
      const int lim = std::min(
          count, static_cast<int>(mhz_table.size()));
      for (int s = 0; s < lim; ++s) {
        const double f = mhz_table[static_cast<std::size_t>(s)];
        if (f <= 0.0) { continue; }   // idle state
        const double r =
            static_cast<double>(api.state_get_residency(ch, s));
        if (r <= 0.0) { continue; }
        wsum += r * f;
        rsum += r;
      }
      if (rsum > 0.0) { result = wsum / rsum; }
    }
    CFRelease(delta);
    return result;
  }

  std::optional<double> sample_temp_() { return smc.read_gpu_temp(); }

  void loop_()
  {
    while (running.load(std::memory_order_relaxed)) {
      if (auto w = sample_power_()) { a_pwr.add(*w); }
      if (auto f = sample_freq_())  { a_freq.add(*f); }
      const GpuInfo g = query_gpu_iokit_();
      if (g.util_ok) { a_util.add(g.util_pct); }
      if (auto t = sample_temp_()) { a_temp.add(*t); }
      if (auto m = process_footprint_mb_()) { a_mem.add(*m); }
      std::this_thread::sleep_for(std::chrono::milliseconds(85));
    }
  }

  void join_()
  {
    running.store(false);
    if (worker.joinable()) { worker.join(); }
  }
};

GpuTelemetrySampler::GpuTelemetrySampler()
  : _impl(std::make_unique<Impl>())
{}

GpuTelemetrySampler::~GpuTelemetrySampler() = default;

void
GpuTelemetrySampler::start()
{
  Impl& im = *_impl;
  if (im.running.load()) { im.join_(); }
  im.a_freq = im.a_temp = im.a_util = im.a_pwr = im.a_mem = Acc{};
  im.thermal.clear();
  // Seed the running deltas with a baseline sample.
  const auto& api = ioreport::resolve();
  if (im.pwr_ready) {
    if (im.pwr_prev) { CFRelease(im.pwr_prev); im.pwr_prev = nullptr; }
    im.pwr_prev = api.create_samples(im.pwr_sub, im.pwr_chans, nullptr);
    im.pwr_t    = std::chrono::steady_clock::now();
    if (!im.pwr_prev) { im.pwr_ready = false; }
  }
  if (im.frq_ready) {
    if (im.frq_prev) { CFRelease(im.frq_prev); im.frq_prev = nullptr; }
    im.frq_prev = api.create_samples(im.frq_sub, im.frq_chans, nullptr);
    if (!im.frq_prev) { im.frq_ready = false; }
  }
  im.running.store(true);
  im.worker = std::thread([&im] { im.loop_(); });
}

GpuTelemetry
GpuTelemetrySampler::stop()
{
  Impl& im = *_impl;
  im.join_();
  GpuTelemetry t;
  t.freq_mhz      = im.a_freq.done();
  t.temp_c        = im.a_temp.done();
  t.util_pct      = im.a_util.done();
  t.power_w       = im.a_pwr.done();
  t.footprint_mb  = im.a_mem.done();
  t.thermal_state = thermal_state_label_();
  return t;
}

std::string
GpuTelemetrySampler::gpu_model()
{
  return query_gpu_iokit_().model;
}

std::uint64_t
GpuTelemetrySampler::gpu_core_count()
{
  return query_gpu_iokit_().core_count;
}

}  // namespace vpipe
