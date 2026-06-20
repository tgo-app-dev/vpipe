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

#include <fstream>
#include <mutex>
#include <string>
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

struct Registry {
  std::mutex mu;
  std::unordered_map<std::string,
                     std::pair<const unsigned char*, std::size_t>> map;
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

bool
MetalCompute::supports_matrix_cores() const noexcept
{
  // GPUFamilyApple10 (Apple M5) is the first with hardware matrix units.
  // Families are cumulative supersets, so supportsFamily(Apple10) is true
  // on M5 and any newer Apple GPU and false on M4 / M3 / earlier.
  if (!_impl->valid || _impl->device == nullptr) { return false; }
  return _impl->device->supportsFamily(MTL::GPUFamilyApple10);
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

  const unsigned char* bytes = nullptr;
  std::size_t          size  = 0;
  if (!_embedded::find_embedded_metallib(name, &bytes, &size)) {
    session()->warn(fmt(
        "MetalCompute::load_library: no embedded metallib registered "
        "as '{}'", name_str));
    return ComputeLibrary{};
  }

  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

  // Wrap the static-const bytes in a dispatch_data_t. Use a no-op
  // destructor block (the registry's bytes outlive the process, so
  // libdispatch must not free them). We hold the dispatch_data_t
  // ref only across the newLibrary call; MTL::Library takes its
  // own internal hold on the bytes it needs.
  dispatch_data_t data = dispatch_data_create(
      bytes, size,
      dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0),
      ^{ /* no-op: bytes are static const, owned by the registry */ });

  NS::Error*    err = nullptr;
  MTL::Library* lib = _impl->device->newLibrary(data, &err);
  dispatch_release(data);

  if (lib == nullptr) {
    const char* msg = "(no description)";
    if (err != nullptr && err->localizedDescription() != nullptr) {
      msg = err->localizedDescription()->utf8String();
    }
    session()->warn(fmt(
        "MetalCompute::load_library: device->newLibrary failed for "
        "'{}': {}", name_str, msg));
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
