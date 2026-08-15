#include "apple-silicon/metal-compute/metal-compute.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

// metal-cpp's MTL::Private selector / class tables (e.g.
// s_kMTLHeapDescriptor) must be emitted in exactly one TU in the dylib.
//   * MLX build (VPIPE_FOUNDATION_FROM_MLX set): libmlx's
//     mlx/backend/metal/device.cpp already emits them -- skip here to
//     avoid duplicate symbols.
//   * no-MLX build: libmlx isn't linked, so emit them here. (The
//     Foundation NS::Private tables come from coreml-private.cc, which
//     defines NS_PRIVATE_IMPLEMENTATION under the same condition.)
// The #define must precede the metal-cpp umbrella include.
#ifndef VPIPE_FOUNDATION_FROM_MLX
#define MTL_PRIVATE_IMPLEMENTATION
#endif
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <sys/sysctl.h>

#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

using namespace std;

namespace MTL {
class BinaryArchive;
class ComputePipelineState;
class Heap;
class ResidencySet;
}

namespace vpipe::metal_compute {

// ---- Embedded metallib registry (process-wide) -------------------
//
// Each add_vpipe_metal_kernel() CMake call emits a generated
// `<name>_embed.cc` TU that calls register_embedded_metallib at
// static-init time. The registry is a Meyers singleton so static-
// init order across TUs doesn't matter.

namespace _embedded {

namespace {

// A kernel is registered as EITHER compiled metallib bytes (build-time
// metallib mode) or MSL source (runtime-compile mode) -- never both for
// the same name. `src` is a NUL-terminated, include-flattened source
// string owned by a static array in the generated TU; `lang` is a
// language-version code (0 = default, 40 = metal4.0) applied via
// MTLCompileOptions when compiling via newLibraryWithSource:.
struct SourceEntry {
  const char* src  = nullptr;
  std::size_t len  = 0;
  int         lang = 0;
};

struct Registry {
  std::mutex mu;
  std::unordered_map<std::string,
                     std::pair<const unsigned char*, std::size_t>> map;
  std::unordered_map<std::string, SourceEntry>                     src_map;
};

Registry& registry()
{
  static Registry r;
  return r;
}

}  // namespace

void
register_embedded_metallib(const char* name,
                           const unsigned char* data,
                           std::size_t size)
{
  Registry& r = registry();
  std::lock_guard<std::mutex> g(r.mu);
  r.map.emplace(std::string(name), std::make_pair(data, size));
}

bool
find_embedded_metallib(std::string_view name,
                       const unsigned char** out_data,
                       std::size_t* out_size)
{
  Registry& r = registry();
  std::lock_guard<std::mutex> g(r.mu);
  auto it = r.map.find(std::string(name));
  if (it == r.map.end()) {
    return false;
  }
  *out_data = it->second.first;
  *out_size = it->second.second;
  return true;
}

// Runtime-compile mode: register the include-flattened MSL source for a
// kernel (emitted by embed-metal-source.cmake). load_library() compiles it
// via newLibraryWithSource: on first use.
void
register_embedded_metal_source(const char* name, const char* src,
                               std::size_t len, int lang_version)
{
  Registry& r = registry();
  std::lock_guard<std::mutex> g(r.mu);
  r.src_map.emplace(std::string(name),
                    SourceEntry{src, len, lang_version});
}

bool
find_embedded_metal_source(std::string_view name, const char** out_src,
                           std::size_t* out_len, int* out_lang)
{
  Registry& r = registry();
  std::lock_guard<std::mutex> g(r.mu);
  auto it = r.src_map.find(std::string(name));
  if (it == r.src_map.end()) {
    return false;
  }
  *out_src  = it->second.src;
  *out_len  = it->second.len;
  *out_lang = it->second.lang;
  return true;
}

namespace {

// Process-lifetime owner for bytes handed in at RUNTIME (a plugin's
// offline-compiled metallib). The build-embedded path points the registry
// at static-const arrays; a runtime caller's buffer has no such lifetime,
// so we copy into here. std::deque never relocates existing elements, so
// pointers into the stored vectors stay stable, and the store is never
// freed (plugins are never dlclose'd), which keeps the no-op dispatch_data
// destructor in load_library valid.
struct RuntimeStore {
  std::mutex mu;
  std::deque<std::vector<unsigned char>> blobs;
};

RuntimeStore& runtime_store()
{
  static RuntimeStore s;
  return s;
}

}  // namespace

// Copy `data` into process-owned storage and register it under `name`.
// First-wins: returns false (a no-op) if `name` is already registered
// (built-in or runtime) or the bytes are empty.
bool
register_runtime_metallib(std::string_view name,
                          const unsigned char* data, std::size_t size)
{
  if (data == nullptr || size == 0) {
    return false;
  }
  const unsigned char* d = nullptr;
  std::size_t s = 0;
  if (find_embedded_metallib(name, &d, &s)) {
    return false;                          // name already taken
  }
  RuntimeStore& store = runtime_store();
  const unsigned char* owned = nullptr;
  {
    std::lock_guard<std::mutex> g(store.mu);
    store.blobs.emplace_back(data, data + size);
    owned = store.blobs.back().data();
  }
  register_embedded_metallib(std::string(name).c_str(), owned, size);
  return true;
}

}  // namespace _embedded

struct MetalCompute::Impl {
  MTL::Device* device = nullptr;
  bool         valid  = false;

  // Cache state. The mutex guards both maps; loads and PSO builds
  // happen outside it (then we take it briefly to publish). The
  // cache holds one refcount on each cached pointer; per-handout
  // ComputeLibrary / ComputeFunction additionally retain.
  mutable std::mutex                                       cache_mu;
  std::unordered_map<std::string, MTL::Library*>           lib_cache;
  std::unordered_map<std::string,
                     MTL::ComputePipelineState*>           pso_cache;

  // Lazily-created automatic-placement Shared-storage heap.
  // Small Tracked allocations (<= kSmallAllocThreshold) sub-
  // allocate from this; on heap-full or non-fit, fall back to
  // device->newBuffer. The heap returns space to itself when sub-
  // allocated buffers are released. Guarded by alloc_mu.
  static constexpr std::size_t kSmallAllocThreshold = 64 * 1024;
  static constexpr std::size_t kHeapSize            = 4 * 1024 * 1024;

  mutable std::mutex   alloc_mu;
  MTL::Heap*           small_heap            = nullptr;
  std::size_t          alloc_buffers_heap    = 0;
  std::size_t          alloc_buffers_device  = 0;

  // PSO binary archive (Metal 3+). Allocated by
  // set_binary_archive_path; nullptr means "archive disabled".
  // Guarded by cache_mu since the PSO build path also touches it.
  MTL::BinaryArchive*  binary_archive            = nullptr;
  std::string          binary_archive_path;
  std::size_t          pso_compiled_with_archive = 0;
  std::size_t          pso_added_to_archive      = 0;

  // Working-set residency tracking (Metal 3+). Lazily allocated
  // on first residency_* call; nullptr means unsupported or
  // allocation failed. Guarded by alloc_mu (cheap reuse rather
  // than introducing a 3rd mutex).
  MTL::ResidencySet*   residency_set      = nullptr;
  bool                 residency_probed   = false;
  std::size_t          residency_add_calls    = 0;
  std::size_t          residency_remove_calls = 0;
};

MetalCompute::MetalCompute(const SessionContextIntf* session)
  : SessionMember(session),
    _impl(make_unique<Impl>())
{
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

  _impl->device = MTL::CreateSystemDefaultDevice();
  if (_impl->device == nullptr) {
    session->warn(fmt(
        "MetalCompute: MTL::CreateSystemDefaultDevice returned null; "
        "the framework is disabled for this session"));
    pool->release();
    return;
  }
  _impl->valid = true;

  // Pre-allocate the small-buffer heap. Failure is non-fatal --
  // we fall back to direct device->newBuffer for every allocation.
  {
    MTL::HeapDescriptor* hd = MTL::HeapDescriptor::alloc()->init();
    hd->setSize(static_cast<NS::UInteger>(Impl::kHeapSize));
    hd->setStorageMode(MTL::StorageModeShared);
    hd->setHazardTrackingMode(MTL::HazardTrackingModeTracked);
    hd->setType(MTL::HeapTypeAutomatic);
    _impl->small_heap = _impl->device->newHeap(hd);
    hd->release();
    if (_impl->small_heap == nullptr) {
      session->warn(fmt(
          "MetalCompute: device->newHeap({} MB) failed; small-buffer "
          "sub-allocation disabled",
          Impl::kHeapSize >> 20));
    }
  }

  pool->release();
}

MetalCompute::~MetalCompute()
{
  // Release everything the caches retain. No need to hold the
  // mutex; ~MetalCompute is racy with any concurrent call by
  // definition (the SessionMember is gone).
  for (auto& [k, pso] : _impl->pso_cache) {
    if (pso != nullptr) {
      pso->release();
    }
  }
  _impl->pso_cache.clear();
  for (auto& [k, lib] : _impl->lib_cache) {
    if (lib != nullptr) {
      lib->release();
    }
  }
  _impl->lib_cache.clear();

  if (_impl->residency_set != nullptr) {
    _impl->residency_set->release();
    _impl->residency_set = nullptr;
  }

  if (_impl->binary_archive != nullptr) {
    _impl->binary_archive->release();
    _impl->binary_archive = nullptr;
  }

  if (_impl->small_heap != nullptr) {
    _impl->small_heap->release();
    _impl->small_heap = nullptr;
  }

  if (_impl->device != nullptr) {
    _impl->device->release();
    _impl->device = nullptr;
  }
}

bool
MetalCompute::valid() const noexcept
{
  return _impl->valid;
}

MTL::Device*
MetalCompute::device() const noexcept
{
  return _impl->device;
}

// Running macOS version as major*100+minor ("26.2.1" -> 2602), or 0 if
// it cannot be read.
static int
macos_version_() noexcept
{
  char   buf[64] = {0};
  size_t n       = sizeof(buf) - 1;
  if (::sysctlbyname("kern.osproductversion", buf, &n, nullptr, 0) != 0) {
    return 0;
  }
  int major = 0, minor = 0;
  if (std::sscanf(buf, "%d.%d", &major, &minor) < 1) { return 0; }
  return major * 100 + minor;
}

bool
MetalCompute::supports_matrix_cores() const noexcept
{
  // GPUFamilyApple10 (Apple M5) is the first with hardware matrix units.
  // Families are cumulative supersets, so supportsFamily(Apple10) is true
  // on M5 and any newer Apple GPU and false on M4 / M3 / earlier.
  if (!_impl->valid || _impl->device == nullptr) { return false; }

  // One switch that takes the WHOLE matrix-core path out, ahead of every
  // other check. The per-family kill-switches (VPIPE_QWEN_NO_MMA,
  // VPIPE_GEMMA_NO_MMA, VPIPE_KREA2_NO_MMA2, ...) each disable one
  // consumer; a GPU hang does not announce which kernel caused it, so
  // bisecting a wedge with them costs a reboot per guess. This is also
  // the workaround to hand someone whose machine hangs: it puts them on
  // the pre-M5 path, which is the one the whole test suite covers.
  static const bool disabled =
    std::getenv("VPIPE_NO_MATRIX_CORES") != nullptr;
  if (disabled) { return false; }

  if (!_impl->device->supportsFamily(MTL::GPUFamilyApple10)) { return false; }

  // ...AND macOS 26.2 or newer. The hardware is not the only condition.
  //
  // The NAX kernels are compiled -mmacosx-version-min=26.2 because below
  // that target the matmul2d codegen is wrong (mlx #3622; see
  // apple-silicon/metal-compute/CMakeLists.txt). Their metallibs
  // therefore record air64_v28-apple-macosx26.2.0, while the rest of the
  // build records the app's own, lower deployment target.
  //
  // Loading 26.2-targeted AIR on an older runtime does NOT fail cleanly.
  // OBSERVED on an M5 running an earlier 26.x: the app started, took the
  // matrix-core path because the family check passed, and hung the GPU
  // hard -- 100% utilisation that outlived the process and needed a
  // reboot. A wrong answer would have been bad; a wedged GPU is worse,
  // and nothing in the process can recover from it.
  //
  // So the OS is part of the capability, not an assumption. Below 26.2
  // every caller keeps the steel/ALU path, which is correct everywhere
  // and merely slower.
  //
  // VPIPE_FORCE_MATRIX_CORES=1 overrides, for bringing the NAX path up
  // on a machine where the mismatch is understood and a reboot is
  // acceptable. It is not a supported configuration.
  static const int  os     = macos_version_();
  static const bool forced =
    std::getenv("VPIPE_FORCE_MATRIX_CORES") != nullptr;
  return forced || os >= 2602;
}

MetalCompute::MatrixCoreGate
MetalCompute::matrix_core_gate() const noexcept
{
  MatrixCoreGate g;
  g.env_disabled = std::getenv("VPIPE_NO_MATRIX_CORES") != nullptr;
  g.env_forced   = std::getenv("VPIPE_FORCE_MATRIX_CORES") != nullptr;
  g.macos        = macos_version_() >= 2602;
  g.device_family = _impl->valid && _impl->device != nullptr &&
                    _impl->device->supportsFamily(MTL::GPUFamilyApple10);
  return g;
}

MetalCompute::AllocStats
MetalCompute::alloc_stats() const noexcept
{
  std::lock_guard<std::mutex> g(_impl->alloc_mu);
  AllocStats out;
  out.buffers_from_heap   = _impl->alloc_buffers_heap;
  out.buffers_from_device = _impl->alloc_buffers_device;
  return out;
}

bool
MetalCompute::set_binary_archive_path(std::string_view path) const
{
  if (!_impl->valid) {
    return false;
  }
  std::lock_guard<std::mutex> g(_impl->cache_mu);

  // Tear down any existing archive (caller intentionally rebinding).
  if (_impl->binary_archive != nullptr) {
    _impl->binary_archive->release();
    _impl->binary_archive = nullptr;
    _impl->binary_archive_path.clear();
  }
  if (path.empty()) {
    return true;
  }

  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  MTL::BinaryArchiveDescriptor* d =
      MTL::BinaryArchiveDescriptor::alloc()->init();
  std::string path_str(path);

  // setUrl(nullptr) creates a fresh in-memory archive; otherwise
  // Metal attempts to load the existing file. If the file doesn't
  // exist or fails to parse, we drop back to a fresh archive
  // rather than disabling the feature.
  std::ifstream probe(path_str, std::ios::binary);
  const bool exists = probe.good();
  probe.close();
  if (exists) {
    NS::String* p = NS::String::string(
        path_str.c_str(), NS::UTF8StringEncoding);
    NS::URL* url = NS::URL::fileURLWithPath(p);
    d->setUrl(url);
  }
  NS::Error* err = nullptr;
  MTL::BinaryArchive* ar = _impl->device->newBinaryArchive(d, &err);
  d->release();

  if (ar == nullptr && exists) {
    // Existing file is unreadable; retry without it.
    MTL::BinaryArchiveDescriptor* d2 =
        MTL::BinaryArchiveDescriptor::alloc()->init();
    err = nullptr;
    ar = _impl->device->newBinaryArchive(d2, &err);
    d2->release();
  }
  if (ar == nullptr) {
    const char* msg = "(no description)";
    if (err != nullptr && err->localizedDescription() != nullptr) {
      msg = err->localizedDescription()->utf8String();
    }
    session()->warn(fmt(
        "MetalCompute::set_binary_archive_path: newBinaryArchive "
        "failed: {}", msg));
    pool->release();
    return false;
  }
  _impl->binary_archive      = ar;
  _impl->binary_archive_path = std::move(path_str);
  pool->release();
  return true;
}

bool
MetalCompute::save_binary_archive() const
{
  if (!_impl->valid) {
    return false;
  }
  std::lock_guard<std::mutex> g(_impl->cache_mu);
  if (_impl->binary_archive == nullptr
      || _impl->binary_archive_path.empty()) {
    return false;
  }
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  NS::String* p = NS::String::string(
      _impl->binary_archive_path.c_str(), NS::UTF8StringEncoding);
  NS::URL* url = NS::URL::fileURLWithPath(p);
  NS::Error* err = nullptr;
  const bool ok = _impl->binary_archive->serializeToURL(url, &err);
  if (!ok) {
    const char* msg = "(no description)";
    if (err != nullptr && err->localizedDescription() != nullptr) {
      msg = err->localizedDescription()->utf8String();
    }
    session()->warn(fmt(
        "MetalCompute::save_binary_archive: serializeToURL failed "
        "for '{}': {}", _impl->binary_archive_path, msg));
  }
  pool->release();
  return ok;
}

MTL::ComputePipelineState*
MetalCompute::build_pso_(MTL::Function* fn, NS::Error** out_err) const
{
  // Fast path: no archive configured -> simplest form.
  if (_impl->binary_archive == nullptr) {
    return _impl->device->newComputePipelineState(fn, out_err);
  }

  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  MTL::ComputePipelineDescriptor* d =
      MTL::ComputePipelineDescriptor::alloc()->init();
  d->setComputeFunction(fn);

  // Cache the binary archive as a single-element NS::Array on the
  // descriptor; Metal will probe the archive for a matching binary
  // before falling back to a compile.
  const MTL::BinaryArchive* archives[1] = { _impl->binary_archive };
  NS::Array* arr =
      NS::Array::array(reinterpret_cast<const NS::Object* const*>(archives),
                       1);
  d->setBinaryArchives(arr);

  MTL::AutoreleasedComputePipelineReflection refl = nullptr;
  MTL::ComputePipelineState* pso =
      _impl->device->newComputePipelineState(
          d, MTL::PipelineOptionNone, &refl, out_err);

  if (pso != nullptr) {
    {
      std::lock_guard<std::mutex> g(_impl->cache_mu);
      ++_impl->pso_compiled_with_archive;
    }
    // Register the freshly-built PSO with the archive so a later
    // save serializes it. A non-fatal failure here just means the
    // archive doesn't grow; the PSO itself is fine.
    NS::Error* add_err = nullptr;
    bool added = _impl->binary_archive
                     ->addComputePipelineFunctions(d, &add_err);
    if (added) {
      std::lock_guard<std::mutex> g(_impl->cache_mu);
      ++_impl->pso_added_to_archive;
    }
  }

  d->release();
  pool->release();
  return pso;
}

// ---- Residency set ----------------------------------------------
//
// `ensure_rs_` is a lambda-ish helper invoked from each
// residency_* method. Caller already holds alloc_mu. Caches the
// probe result so we don't repeatedly retry newResidencySet on
// older OS / pre-Metal3 GPUs.
//
// (We can't put it in an anonymous namespace because Impl is a
// private nested type. Use a static lambda assigned per use site
// is overkill; just open-code the lookup at each call.)

#define VPIPE_MC_ENSURE_RS(impl)                                  \
  ([&]() -> MTL::ResidencySet* {                                  \
    if ((impl)->residency_probed) {                               \
      return (impl)->residency_set;                               \
    }                                                             \
    (impl)->residency_probed = true;                              \
    NS::AutoreleasePool* p = NS::AutoreleasePool::alloc()->init();\
    MTL::ResidencySetDescriptor* d =                              \
        MTL::ResidencySetDescriptor::alloc()->init();             \
    d->setInitialCapacity(64);                                    \
    NS::Error* e = nullptr;                                       \
    (impl)->residency_set = (impl)->device->newResidencySet(d,&e);\
    d->release();                                                 \
    p->release();                                                 \
    return (impl)->residency_set;                                 \
  })()

bool
MetalCompute::residency_set_supported() const noexcept
{
  if (!_impl->valid) {
    return false;
  }
  std::lock_guard<std::mutex> g(_impl->alloc_mu);
  // Probe forces the lazy creation; subsequent calls are O(1).
  return VPIPE_MC_ENSURE_RS(_impl.get()) != nullptr;
}

bool
MetalCompute::residency_add(const SharedBuffer& b) const
{
  if (!_impl->valid || b.empty()) {
    return false;
  }
  std::lock_guard<std::mutex> g(_impl->alloc_mu);
  MTL::ResidencySet* rs = VPIPE_MC_ENSURE_RS(_impl.get());
  if (rs == nullptr) {
    return false;
  }
  rs->addAllocation(reinterpret_cast<MTL::Allocation*>(b.mtl_buffer()));
  ++_impl->residency_add_calls;
  return true;
}

bool
MetalCompute::residency_add(const Texture& t) const
{
  if (!_impl->valid || !t.valid()) {
    return false;
  }
  std::lock_guard<std::mutex> g(_impl->alloc_mu);
  MTL::ResidencySet* rs = VPIPE_MC_ENSURE_RS(_impl.get());
  if (rs == nullptr) {
    return false;
  }
  rs->addAllocation(reinterpret_cast<MTL::Allocation*>(t.mtl_texture()));
  ++_impl->residency_add_calls;
  return true;
}

bool
MetalCompute::residency_remove(const SharedBuffer& b) const
{
  if (!_impl->valid || b.empty()) {
    return false;
  }
  std::lock_guard<std::mutex> g(_impl->alloc_mu);
  if (_impl->residency_set == nullptr) {
    return false;
  }
  _impl->residency_set->removeAllocation(
      reinterpret_cast<MTL::Allocation*>(b.mtl_buffer()));
  ++_impl->residency_remove_calls;
  return true;
}

bool
MetalCompute::residency_remove(const Texture& t) const
{
  if (!_impl->valid || !t.valid()) {
    return false;
  }
  std::lock_guard<std::mutex> g(_impl->alloc_mu);
  if (_impl->residency_set == nullptr) {
    return false;
  }
  _impl->residency_set->removeAllocation(
      reinterpret_cast<MTL::Allocation*>(t.mtl_texture()));
  ++_impl->residency_remove_calls;
  return true;
}

bool
MetalCompute::residency_commit() const
{
  if (!_impl->valid) {
    return false;
  }
  std::lock_guard<std::mutex> g(_impl->alloc_mu);
  if (_impl->residency_set == nullptr) {
    return false;
  }
  _impl->residency_set->commit();
  return true;
}

bool
MetalCompute::residency_request() const
{
  if (!_impl->valid) {
    return false;
  }
  std::lock_guard<std::mutex> g(_impl->alloc_mu);
  if (_impl->residency_set == nullptr) {
    return false;
  }
  _impl->residency_set->requestResidency();
  return true;
}

bool
MetalCompute::residency_end() const
{
  if (!_impl->valid) {
    return false;
  }
  std::lock_guard<std::mutex> g(_impl->alloc_mu);
  if (_impl->residency_set == nullptr) {
    return false;
  }
  _impl->residency_set->endResidency();
  return true;
}

MetalCompute::ResidencyStats
MetalCompute::residency_stats() const noexcept
{
  std::lock_guard<std::mutex> g(_impl->alloc_mu);
  ResidencyStats out;
  out.add_calls    = _impl->residency_add_calls;
  out.remove_calls = _impl->residency_remove_calls;
  out.current      = _impl->residency_set != nullptr
                         ? _impl->residency_set->allocationCount()
                         : 0;
  return out;
}

MetalCompute::MemoryBudget
MetalCompute::memory_budget() const noexcept
{
  MemoryBudget out;
  if (!_impl->valid || _impl->device == nullptr) {
    return out;
  }
  out.recommended =
      static_cast<std::size_t>(_impl->device->recommendedMaxWorkingSetSize());
  out.allocated =
      static_cast<std::size_t>(_impl->device->currentAllocatedSize());
  out.headroom = out.recommended > out.allocated
                     ? out.recommended - out.allocated
                     : 0;
  // Reclaimable physical memory: free + purgeable + file-backed(external)
  // pages. These are exactly the pages the OS can hand a new allocation --
  // including clean mmap'd weight pages (external), which is why a large decode
  // can succeed alongside mmap'd model weights. Conservative: dirty inactive
  // pages (may need swap) are excluded.
  {
    vm_statistics64_data_t vm;
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vm), &n)
        == KERN_SUCCESS) {
      vm_size_t page = 0;
      if (host_page_size(mach_host_self(), &page) != KERN_SUCCESS) {
        page = 16384;                       // Apple Silicon default
      }
      const std::size_t pages = (std::size_t)vm.free_count
                              + (std::size_t)vm.purgeable_count
                              + (std::size_t)vm.external_page_count;
      out.available_physical = pages * (std::size_t)page;
      // The same query, without the file cache: what a DURABLE allocation
      // can take without turning reclaimable pages into anonymous ones.
      // Speculative pages are read-ahead the OS drops for free, so they
      // count; external pages do not.
      std::size_t idle = (std::size_t)vm.free_count
                       + (std::size_t)vm.purgeable_count
                       + (std::size_t)vm.speculative_count;
      // macOS deliberately keeps `free` small -- spare RAM becomes file
      // cache -- so a box with a large COLD cache (someone else's file
      // activity, not this model's re-reads) reads as having no room and
      // would under-grow. VPIPE_RESIDENCY_CACHE_PCT lets a machine count
      // that fraction of the cache as growable. It defaults to 0 because
      // the two cases are not distinguishable from here and the costs are
      // not symmetric: under-growing streams a few more blocks, while
      // over-growing is the swap storm this figure exists to prevent.
      static const int cache_pct = [] {
        const char* e = std::getenv("VPIPE_RESIDENCY_CACHE_PCT");
        if (e == nullptr) { return 0; }
        const int v = std::atoi(e);
        return v < 0 ? 0 : (v > 100 ? 100 : v);
      }();
      if (cache_pct > 0) {
        idle += (std::size_t)vm.external_page_count * (std::size_t)cache_pct
                / 100;
      }
      out.free_physical = idle * (std::size_t)page;
      out.compressed =
          (std::size_t)vm.compressor_page_count * (std::size_t)page;
    }
  }
  {
    task_vm_info_data_t ti;
    mach_msg_type_number_t c = TASK_VM_INFO_COUNT;
    if (::task_info(mach_task_self(), TASK_VM_INFO,
                    reinterpret_cast<task_info_t>(&ti), &c) == KERN_SUCCESS) {
      out.self_compressed = (std::size_t)ti.compressed;
      out.self_footprint  = (std::size_t)ti.phys_footprint;
    }
  }
  {
    struct xsw_usage sw;
    std::size_t n = sizeof(sw);
    if (::sysctlbyname("vm.swapusage", &sw, &n, nullptr, 0) == 0) {
      out.swap_used = (std::size_t)sw.xsu_used;
    }
  }
  // Mirrors model_memory::phys_ram(): the planning phase sizes the box
  // against VPIPE_RAM_LIMIT_MB when it is set, and a live check that
  // disagreed would undo exactly the bounded-box configuration the
  // variable exists to make reproducible.
  {
    if (const char* e = std::getenv("VPIPE_RAM_LIMIT_MB")) {
      const long long mb = std::atoll(e);
      if (mb > 0) { out.total_physical = (std::size_t)mb << 20; }
    }
    if (out.total_physical == 0) {
      std::uint64_t mem = 0;
      std::size_t n = sizeof(mem);
      if (::sysctlbyname("hw.memsize", &mem, &n, nullptr, 0) == 0) {
        out.total_physical = (std::size_t)mem;
      }
    }
  }
  return out;
}

MetalCompute::PsoArchiveStats
MetalCompute::pso_archive_stats() const noexcept
{
  std::lock_guard<std::mutex> g(_impl->cache_mu);
  PsoArchiveStats out;
  out.compiled_with_archive_set = _impl->pso_compiled_with_archive;
  out.added_to_archive          = _impl->pso_added_to_archive;
  return out;
}

SharedBuffer
MetalCompute::make_shared_buffer(std::size_t byte_size,
                                 std::size_t alignment,
                                 HazardTracking ht) const
{
  if (!_impl->valid || byte_size == 0) {
    return SharedBuffer{};
  }

  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

  MTL::ResourceOptions opts = MTL::ResourceStorageModeShared;
  if (ht == HazardTracking::Untracked) {
    opts = static_cast<MTL::ResourceOptions>(
        opts | MTL::ResourceHazardTrackingModeUntracked);
  }

  MTL::Buffer* buf = nullptr;

  // Heap fast path: small Tracked allocations sub-allocate from
  // the pre-created automatic heap. Untracked bypasses (heap's
  // hazard mode is fixed at creation), and large requests bypass
  // because the heap's total capacity is fixed.
  const bool can_use_heap =
      _impl->small_heap != nullptr
      && ht == HazardTracking::Tracked
      && byte_size <= Impl::kSmallAllocThreshold;
  if (can_use_heap) {
    std::lock_guard<std::mutex> g(_impl->alloc_mu);
    buf = _impl->small_heap->newBuffer(
        static_cast<NS::UInteger>(byte_size), opts);
    if (buf != nullptr) {
      ++_impl->alloc_buffers_heap;
    }
  }
  if (buf == nullptr) {
    buf = _impl->device->newBuffer(
        static_cast<NS::UInteger>(byte_size), opts);
    if (buf != nullptr) {
      std::lock_guard<std::mutex> g(_impl->alloc_mu);
      ++_impl->alloc_buffers_device;
    }
  }
  if (buf == nullptr) {
    session()->warn(fmt(
        "MetalCompute::make_shared_buffer: allocation failed "
        "(byte_size={})", byte_size));
    pool->release();
    return SharedBuffer{};
  }
  void* contents = buf->contents();

  // MTL Shared buffers are page-aligned, so any alignment <= page
  // size is satisfied automatically. Warn rather than fail loudly
  // if the caller asked for something tighter than what the buffer
  // can promise; in practice MTL satisfies <= 16384 byte alignment.
  if (alignment != 0 &&
      (reinterpret_cast<std::uintptr_t>(contents)
       & (alignment - 1)) != 0) {
    session()->warn(fmt(
        "MetalCompute::make_shared_buffer: contents pointer {} is "
        "not aligned to requested {} bytes", contents, alignment));
  }

  SharedBuffer out(buf, contents, byte_size);
  // Put it on the books: this is the owning handle (subviews of it share the
  // MTL::Buffer by refcount and are deliberately not counted again).
  out._accounted = true;
  account_alloc_(byte_size);
  pool->release();
  return out;
}

SharedBuffer
MetalCompute::make_no_copy_buffer(void* ptr, std::size_t byte_size) const
{
  if (!_impl->valid || ptr == nullptr || byte_size == 0) {
    return SharedBuffer{};
  }
  const std::size_t page =
      static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
  // newBufferWithBytesNoCopy requires a page-aligned base; mmap/vm_allocate
  // memory satisfies this. Reject anything else rather than silently copy.
  if ((reinterpret_cast<std::uintptr_t>(ptr) & (page - 1)) != 0) {
    return SharedBuffer{};
  }
  // Metal wants the length page-aligned too; round up. Accessing the tail of
  // the last page is safe for an mmap (zero-filled past EOF).
  const std::size_t mapped_len = (byte_size + page - 1) & ~(page - 1);

  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  // Null deallocator: Metal does not own or free the memory.
  MTL::Buffer* buf = _impl->device->newBuffer(
      ptr, static_cast<NS::UInteger>(mapped_len),
      MTL::ResourceStorageModeShared, nullptr);
  if (buf == nullptr) {
    session()->warn(fmt(
        "MetalCompute::make_no_copy_buffer: newBufferWithBytesNoCopy failed "
        "(byte_size={})", byte_size));
    pool->release();
    return SharedBuffer{};
  }
  {
    std::lock_guard<std::mutex> g(_impl->alloc_mu);
    ++_impl->alloc_buffers_device;
  }
  // Report the caller's logical size (not the page-rounded length).
  SharedBuffer out = SharedBuffer::wrap(buf, ptr, byte_size);
  pool->release();
  return out;
}

ComputeLibrary
MetalCompute::load_library(std::string_view name) const
{
  if (!_impl->valid) {
    return ComputeLibrary{};
  }
  std::string name_str(name);

  // Fast path: cache hit. Retain once for the handed-out instance.
  {
    std::lock_guard<std::mutex> g(_impl->cache_mu);
    auto it = _impl->lib_cache.find(name_str);
    if (it != _impl->lib_cache.end()) {
      MTL::Library* cached = it->second;
      cached->retain();
      return ComputeLibrary{cached, name_str, this};
    }
  }

  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  NS::Error*           err  = nullptr;
  MTL::Library*        lib  = nullptr;

  const unsigned char* bytes  = nullptr;
  std::size_t          size   = 0;
  const char*          src    = nullptr;
  std::size_t          srclen = 0;
  int                  lang   = 0;

  if (_embedded::find_embedded_metallib(name, &bytes, &size)) {
    // Build-time metallib mode: load the compiled bytes. Wrap the
    // static-const bytes in a dispatch_data_t with a no-op destructor
    // block (the registry's bytes outlive the process, so libdispatch
    // must not free them). We hold the dispatch_data_t ref only across
    // the newLibrary call; MTL::Library takes its own internal hold.
    dispatch_data_t data = dispatch_data_create(
        bytes, size,
        dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0),
        ^{ /* no-op: bytes are static const, owned by the registry */ });
    lib = _impl->device->newLibrary(data, &err);
    dispatch_release(data);
  } else if (_embedded::find_embedded_metal_source(name, &src, &srclen,
                                                   &lang)) {
    // Runtime-compile mode (no build-time Metal toolchain): compile the
    // embedded, include-flattened MSL source via newLibraryWithSource:.
    // Match the AOT flags that affect numerics -- safe math (the offline
    // path uses -fno-fast-math) -- and set the language version for the
    // metal4.0 matrix-core kernels. Result is cached below like any lib.
    (void)srclen;   // src is NUL-terminated; length kept for diagnostics
    NS::String* nssrc = NS::String::string(src, NS::UTF8StringEncoding);
    MTL::CompileOptions* opts = MTL::CompileOptions::alloc()->init();
    opts->setFastMathEnabled(false);
    if (lang >= 40) {
      opts->setLanguageVersion(MTL::LanguageVersion4_0);
    } else if (lang >= 32) {
      opts->setLanguageVersion(MTL::LanguageVersion3_2);
    } else if (lang >= 31) {
      opts->setLanguageVersion(MTL::LanguageVersion3_1);
    } else if (lang >= 30) {
      opts->setLanguageVersion(MTL::LanguageVersion3_0);
    }
    lib = _impl->device->newLibrary(nssrc, opts, &err);
    opts->release();
  } else {
    session()->warn(fmt(
        "MetalCompute::load_library: no embedded metallib or source "
        "registered as '{}'", name_str));
    pool->release();
    return ComputeLibrary{};
  }

  if (lib == nullptr) {
    const char* msg = "(no description)";
    if (err != nullptr && err->localizedDescription() != nullptr) {
      msg = err->localizedDescription()->utf8String();
    }
    session()->warn(fmt(
        "MetalCompute::load_library: newLibrary failed for '{}': {}",
        name_str, msg));
    pool->release();
    return ComputeLibrary{};
  }

  // Publish to cache (keeps one retain). If a concurrent load won
  // the race, release ours and use the cached instance.
  {
    std::lock_guard<std::mutex> g(_impl->cache_mu);
    auto [it, inserted] = _impl->lib_cache.try_emplace(name_str, lib);
    if (!inserted) {
      lib->release();
      lib = it->second;
    }
    lib->retain();  // for the handed-out ComputeLibrary
  }

  ComputeLibrary out(lib, std::move(name_str), this);
  pool->release();
  return out;
}

bool
MetalCompute::register_metal_library(std::string_view     name,
                                     const unsigned char* bytes,
                                     std::size_t          n) const
{
  if (_embedded::register_runtime_metallib(name, bytes, n)) {
    if (session() != nullptr) {
      session()->log_normal(fmt(
          "MetalCompute: registered runtime metal library '{}' ({} bytes)",
          name, n));
    }
    return true;
  }
  if (session() != nullptr) {
    session()->warn(fmt(
        "MetalCompute::register_metal_library: '{}' already registered or "
        "empty; ignored (runtime kernels cannot shadow a built-in)", name));
  }
  return false;
}

bool
MetalCompute::register_metal_library_file(std::string_view name,
                                          std::string_view path) const
{
  std::ifstream in(std::string(path), std::ios::binary);
  if (!in) {
    if (session() != nullptr) {
      session()->warn(fmt(
          "MetalCompute::register_metal_library_file: cannot open '{}'",
          path));
    }
    return false;
  }
  const std::vector<unsigned char> bytes(
      (std::istreambuf_iterator<char>(in)),
      std::istreambuf_iterator<char>());
  return register_metal_library(name, bytes.data(), bytes.size());
}

ComputeFunction
MetalCompute::function(const ComputeLibrary& lib,
                       std::string_view fn_name) const
{
  if (!_impl->valid || !lib.valid()) {
    return ComputeFunction{};
  }
  // Cache key bundles library and function name; the \x01 separator
  // is illegal in MSL identifiers so collisions are impossible.
  std::string pso_key;
  pso_key.reserve(lib.name().size() + fn_name.size() + 1);
  pso_key.append(lib.name());
  pso_key.push_back('\x01');
  pso_key.append(fn_name);

  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

  // Resolve the Function symbol first; this is cheap (no
  // pipeline compile) and tells us whether the name is even valid.
  std::string fn_name_str(fn_name);
  NS::String* ns_name = NS::String::string(
      fn_name_str.c_str(), NS::UTF8StringEncoding);
  MTL::Function* fn = lib.mtl_library()->newFunction(ns_name);
  if (fn == nullptr) {
    pool->release();
    return ComputeFunction{};
  }

  // PSO fast path: cache hit returns retained refcounts.
  {
    std::lock_guard<std::mutex> g(_impl->cache_mu);
    auto it = _impl->pso_cache.find(pso_key);
    if (it != _impl->pso_cache.end()) {
      MTL::ComputePipelineState* cached_pso = it->second;
      cached_pso->retain();
      pool->release();
      return ComputeFunction{fn, cached_pso};
    }
  }

  // Slow path: build the PSO outside the lock (newComputePipeline-
  // State can take meaningful time -- multi-ms compile on first
  // hit -- so don't serialize concurrent unique builds).
  NS::Error* err = nullptr;
  MTL::ComputePipelineState* pso = build_pso_(fn, &err);
  if (pso == nullptr) {
    const char* msg = "(no description)";
    if (err != nullptr && err->localizedDescription() != nullptr) {
      msg = err->localizedDescription()->utf8String();
    }
    session()->warn(fmt(
        "MetalCompute::function: newComputePipelineState failed for "
        "'{}::{}': {}",
        std::string(lib.name()), fn_name_str, msg));
    fn->release();
    pool->release();
    return ComputeFunction{};
  }

  // Publish to cache. If we lost a race with another thread, drop
  // our copy and use the cached one. Then retain for the handout.
  {
    std::lock_guard<std::mutex> g(_impl->cache_mu);
    auto [it, inserted] = _impl->pso_cache.try_emplace(pso_key, pso);
    if (!inserted) {
      pso->release();
      pso = it->second;
    }
    pso->retain();
  }

  pool->release();
  return ComputeFunction{fn, pso};
}

ComputeFunction
MetalCompute::function(const ComputeLibrary& lib,
                       std::string_view fn_name,
                       const FunctionConstants& constants) const
{
  if (!_impl->valid || !lib.valid()) {
    return ComputeFunction{};
  }
  // Cache key: lib + "\x01" + fn + "\x01" + constants signature.
  std::string pso_key;
  const std::string& sig = constants.signature();
  pso_key.reserve(lib.name().size() + fn_name.size() + sig.size() + 2);
  pso_key.append(lib.name());
  pso_key.push_back('\x01');
  pso_key.append(fn_name);
  pso_key.push_back('\x01');
  pso_key.append(sig);

  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

  // Specialize the MTL::Function with the constants.
  std::string fn_name_str(fn_name);
  NS::String* ns_name = NS::String::string(
      fn_name_str.c_str(), NS::UTF8StringEncoding);
  MTL::FunctionConstantValues* fcv = constants.build_mtl_();
  NS::Error* fn_err = nullptr;
  MTL::Function* fn =
      lib.mtl_library()->newFunction(ns_name, fcv, &fn_err);
  fcv->release();
  if (fn == nullptr) {
    const char* msg = "(no description)";
    if (fn_err != nullptr && fn_err->localizedDescription() != nullptr) {
      msg = fn_err->localizedDescription()->utf8String();
    }
    session()->warn(fmt(
        "MetalCompute::function (specialized): newFunction failed "
        "for '{}::{}': {}",
        std::string(lib.name()), fn_name_str, msg));
    pool->release();
    return ComputeFunction{};
  }

  // PSO fast path.
  {
    std::lock_guard<std::mutex> g(_impl->cache_mu);
    auto it = _impl->pso_cache.find(pso_key);
    if (it != _impl->pso_cache.end()) {
      MTL::ComputePipelineState* cached_pso = it->second;
      cached_pso->retain();
      pool->release();
      return ComputeFunction{fn, cached_pso};
    }
  }

  // Slow path: build PSO outside the lock.
  NS::Error* err = nullptr;
  MTL::ComputePipelineState* pso = build_pso_(fn, &err);
  if (pso == nullptr) {
    const char* msg = "(no description)";
    if (err != nullptr && err->localizedDescription() != nullptr) {
      msg = err->localizedDescription()->utf8String();
    }
    session()->warn(fmt(
        "MetalCompute::function (specialized): "
        "newComputePipelineState failed for '{}::{}': {}",
        std::string(lib.name()), fn_name_str, msg));
    fn->release();
    pool->release();
    return ComputeFunction{};
  }

  {
    std::lock_guard<std::mutex> g(_impl->cache_mu);
    auto [it, inserted] = _impl->pso_cache.try_emplace(pso_key, pso);
    if (!inserted) {
      pso->release();
      pso = it->second;
    }
    pso->retain();
  }

  pool->release();
  return ComputeFunction{fn, pso};
}

CommandStream
MetalCompute::make_command_stream() const
{
  if (!_impl->valid) {
    return CommandStream{};
  }
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  MTL::CommandQueue* queue = _impl->device->newCommandQueue();
  if (queue == nullptr) {
    session()->warn(fmt(
        "MetalCompute::make_command_stream: device->newCommandQueue "
        "returned null"));
    pool->release();
    return CommandStream{};
  }
  CommandStream out(queue);
  pool->release();
  return out;
}

Event
MetalCompute::make_event() const
{
  if (!_impl->valid) {
    return Event{};
  }
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  MTL::SharedEvent* ev = _impl->device->newSharedEvent();
  if (ev == nullptr) {
    session()->warn(fmt(
        "MetalCompute::make_event: device->newSharedEvent "
        "returned null"));
    pool->release();
    return Event{};
  }
  Event out(ev);
  pool->release();
  return out;
}

Fence
MetalCompute::make_fence() const
{
  if (!_impl->valid) {
    return Fence{};
  }
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  MTL::Fence* f = _impl->device->newFence();
  if (f == nullptr) {
    session()->warn(fmt(
        "MetalCompute::make_fence: device->newFence returned null"));
    pool->release();
    return Fence{};
  }
  Fence out(f);
  pool->release();
  return out;
}

namespace {

MTL::PixelFormat
to_mtl_pixel_format_for_make_(PixelFormat f) noexcept
{
  switch (f) {
    case PixelFormat::R8Unorm:     return MTL::PixelFormatR8Unorm;
    case PixelFormat::RGBA8Unorm:  return MTL::PixelFormatRGBA8Unorm;
    case PixelFormat::BGRA8Unorm:  return MTL::PixelFormatBGRA8Unorm;
    case PixelFormat::R32Float:    return MTL::PixelFormatR32Float;
    case PixelFormat::RGBA32Float: return MTL::PixelFormatRGBA32Float;
    case PixelFormat::R16Float:    return MTL::PixelFormatR16Float;
    case PixelFormat::RGBA16Float: return MTL::PixelFormatRGBA16Float;
    case PixelFormat::Unknown:     return MTL::PixelFormatInvalid;
  }
  return MTL::PixelFormatInvalid;
}

MTL::TextureUsage
to_mtl_usage_(TextureUsage u) noexcept
{
  unsigned bits = 0;
  if (static_cast<std::uint8_t>(u) & 1) {
    bits |= MTL::TextureUsageShaderRead;
  }
  if (static_cast<std::uint8_t>(u) & 2) {
    bits |= MTL::TextureUsageShaderWrite;
  }
  return static_cast<MTL::TextureUsage>(bits);
}

}  // namespace

Texture
MetalCompute::make_texture(const TextureDesc& desc) const
{
  if (!_impl->valid || desc.width == 0 || desc.height == 0) {
    return Texture{};
  }
  const MTL::PixelFormat mtl_fmt = to_mtl_pixel_format_for_make_(desc.format);
  if (mtl_fmt == MTL::PixelFormatInvalid) {
    session()->warn(fmt(
        "MetalCompute::make_texture: unsupported PixelFormat "
        "ordinal {}",
        static_cast<int>(desc.format)));
    return Texture{};
  }
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  MTL::TextureDescriptor* td = MTL::TextureDescriptor::texture2DDescriptor(
      mtl_fmt,
      static_cast<NS::UInteger>(desc.width),
      static_cast<NS::UInteger>(desc.height),
      /*mipmapped=*/false);
  td->setUsage(to_mtl_usage_(desc.usage));
  td->setStorageMode(desc.storage_mode == TextureStorage::Private
                         ? MTL::StorageModePrivate
                         : MTL::StorageModeShared);
  MTL::Texture* mt = _impl->device->newTexture(td);
  if (mt == nullptr) {
    session()->warn(fmt(
        "MetalCompute::make_texture: device->newTexture returned "
        "null (width={} height={})", desc.width, desc.height));
    pool->release();
    return Texture{};
  }
  Texture out(mt, desc.format, /*cv_handle=*/nullptr);
  pool->release();
  return out;
}

// CV bridge impl lives in texture-cv-bridge.mm so this TU stays in
// pure C++ (CVMetalTextureCache's id<MTLDevice> argument needs the
// Obj-C compiler). The .mm exports flat-C entry points we forward
// to here.
namespace _texture_cv {
extern void* create_texture_from_cv_pixel_buffer(
    MTL::Device* device,
    void* cv_pixel_buffer,
    PixelFormat format,
    MTL::Texture** out_texture);

extern bool create_nv12_textures_from_cv_pixel_buffer(
    MTL::Device* device,
    void* cv_pixel_buffer,
    MTL::Texture** out_luma_tex,
    void** out_luma_cv_handle,
    MTL::Texture** out_chroma_tex,
    void** out_chroma_cv_handle);
}

Texture
MetalCompute::texture_from_cv_pixel_buffer(void* cv_pixel_buffer,
                                           PixelFormat format) const
{
  if (!_impl->valid || cv_pixel_buffer == nullptr) {
    return Texture{};
  }
  MTL::Texture* mt = nullptr;
  void* cv_handle =
      _texture_cv::create_texture_from_cv_pixel_buffer(
          _impl->device, cv_pixel_buffer, format, &mt);
  if (cv_handle == nullptr || mt == nullptr) {
    return Texture{};
  }
  // Texture's destructor releases the CVMetalTextureRef which in
  // turn owns the MTL::Texture lifetime; the friend constructor
  // matches that protocol.
  return Texture(mt, format, cv_handle);
}

YuvBiplanarTextures
MetalCompute::nv12_textures_from_cv_pixel_buffer(
    void* cv_pixel_buffer) const
{
  YuvBiplanarTextures out;
  if (!_impl->valid || cv_pixel_buffer == nullptr) {
    return out;
  }
  MTL::Texture* luma_tex   = nullptr;
  MTL::Texture* chroma_tex = nullptr;
  void* luma_h   = nullptr;
  void* chroma_h = nullptr;
  bool ok = _texture_cv::create_nv12_textures_from_cv_pixel_buffer(
      _impl->device, cv_pixel_buffer,
      &luma_tex, &luma_h, &chroma_tex, &chroma_h);
  if (!ok) {
    return out;
  }
  out.luma   = Texture(luma_tex,   PixelFormat::R8Unorm,  luma_h);
  out.chroma = Texture(chroma_tex, PixelFormat::Unknown,  chroma_h);
  // chroma has PixelFormat RG8Unorm but our public enum doesn't
  // expose that variant; leave format=Unknown so callers don't
  // misread it. The Texture::mtl_texture() pointer is still
  // bindable to a kernel that declares `texture2d<float, ...>`.
  return out;
}

}  // namespace vpipe::metal_compute
