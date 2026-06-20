#ifndef VPIPE_APPLE_SILICON_METAL_COMPUTE_COMPUTE_LIBRARY_H
#define VPIPE_APPLE_SILICON_METAL_COMPUTE_COMPUTE_LIBRARY_H

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace MTL {
class ComputePipelineState;
class Function;
class FunctionConstantValues;
class Library;
}

namespace vpipe::metal_compute {

class MetalCompute;

// Typed bag of MSL `[[function_constant(N)]]` values that pin the
// kernel at JIT-compile time. Hand to ComputeLibrary::function() to
// produce a specialized PSO; the cache keys on (lib_name, fn_name,
// constants signature) so the same constants reuse the same PSO.
//
// MSL declares slots as
//   constant T kFoo [[function_constant(N)]];
// where T is one of bool / int / uint / float (matching the
// available typed setters here).
class FunctionConstants {
public:
  FunctionConstants() noexcept = default;

  FunctionConstants& set_bool (unsigned index, bool v);
  FunctionConstants& set_int  (unsigned index, std::int32_t  v);
  FunctionConstants& set_uint (unsigned index, std::uint32_t v);
  FunctionConstants& set_float(unsigned index, float v);

  bool empty() const noexcept { return _entries.empty(); }

  // Stable string signature suitable as part of a cache key. The
  // representation is internal; callers should NOT parse it.
  const std::string& signature() const;

private:
  friend class MetalCompute;

  enum class DT : std::uint8_t { Bool, Int, UInt, Float };

  struct Entry {
    unsigned       index;
    DT             dtype;
    std::uint32_t  value;   // bit pattern; widest type is uint/float
  };

  // Build an MTL::FunctionConstantValues from this bag. Caller owns
  // the returned object (+1 retain) and must release it.
  MTL::FunctionConstantValues* build_mtl_() const;

  // Insert-or-replace at `index`; invalidates the cached signature.
  void put_(Entry e);

  std::vector<Entry>  _entries;
  mutable std::string _signature_cache;
};

// One Metal kernel entry point (the symbol after `kernel void ...`
// in MSL) plus its compiled compute pipeline state. valid() iff the
// function name resolved AND the PSO was built. The PSO is cached
// on MetalCompute by (library_name, kernel_name); each retrieval is
// independently refcounted.
class ComputeFunction {
public:
  ComputeFunction() noexcept = default;
  ComputeFunction(ComputeFunction&&) noexcept;
  ComputeFunction& operator=(ComputeFunction&&) noexcept;
  ComputeFunction(const ComputeFunction&)            = delete;
  ComputeFunction& operator=(const ComputeFunction&) = delete;
  ~ComputeFunction();

  bool valid() const noexcept { return _pso != nullptr; }

  // Maximum threads in a single threadgroup. Determined by the
  // kernel's register pressure and threadgroup-memory use. Zero on
  // an invalid ComputeFunction.
  unsigned max_total_threads_per_threadgroup() const noexcept;

  // Native SIMD-group ("warp") width for this kernel on the
  // device. Threadgroup sizes that are multiples of this run more
  // efficiently. Zero on an invalid ComputeFunction.
  unsigned thread_execution_width() const noexcept;

  // Threadgroup memory bytes the kernel allocates statically (i.e.
  // declared with `threadgroup T x[N]` in MSL). Excludes any bytes
  // bound dynamically via ComputeEncoder::set_threadgroup_memory_
  // length(). Zero on an invalid ComputeFunction.
  unsigned static_threadgroup_memory_length() const noexcept;

  MTL::Function*             mtl_function() const noexcept { return _fn; }
  MTL::ComputePipelineState* mtl_pso()      const noexcept { return _pso; }

private:
  friend class MetalCompute;
  ComputeFunction(MTL::Function* fn,
                  MTL::ComputePipelineState* pso) noexcept;

  MTL::Function*             _fn  = nullptr;
  MTL::ComputePipelineState* _pso = nullptr;
};

// One loaded `.metallib` worth of compute kernels. Holds a refcount
// on the underlying MTL::Library. Move-only.
//
// Constructed only by MetalCompute::load_library(); the default
// constructor is the empty/moved-from sentinel. MetalCompute caches
// libraries by name, so repeated load_library() calls return
// instances that share the same MTL::Library* under the hood.
class ComputeLibrary {
public:
  ComputeLibrary() noexcept = default;
  ComputeLibrary(ComputeLibrary&&) noexcept;
  ComputeLibrary& operator=(ComputeLibrary&&) noexcept;
  ComputeLibrary(const ComputeLibrary&)            = delete;
  ComputeLibrary& operator=(const ComputeLibrary&) = delete;
  ~ComputeLibrary();

  bool             valid() const noexcept { return _lib != nullptr; }
  std::string_view name()  const noexcept { return _name; }

  // Look up a kernel entry point by its MSL name. Delegates to the
  // owning MetalCompute's PSO cache, so repeated calls return
  // ComputeFunctions backed by the same PSO. Returns an invalid
  // ComputeFunction if the name is unknown or PSO construction
  // failed.
  ComputeFunction function(std::string_view kernel_name) const;

  // Look up a kernel entry point and specialize it at JIT time
  // against `constants`. Each unique (kernel_name, constants
  // signature) gets its own PSO; repeated calls with equal
  // constants hit the cache.
  ComputeFunction function(std::string_view kernel_name,
                           const FunctionConstants& constants) const;

  MTL::Library* mtl_library() const noexcept { return _lib; }

private:
  friend class MetalCompute;
  ComputeLibrary(MTL::Library* lib, std::string name,
                 const MetalCompute* mc) noexcept;

  MTL::Library*       _lib = nullptr;
  std::string         _name;
  const MetalCompute* _mc  = nullptr;
};

}  // namespace vpipe::metal_compute

#endif
