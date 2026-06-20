#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <algorithm>
#include <cstring>
#include <utility>

namespace vpipe::metal_compute {

// ---- FunctionConstants ------------------------------------------

void
FunctionConstants::put_(Entry e)
{
  _signature_cache.clear();
  for (auto& cur : _entries) {
    if (cur.index == e.index) {
      cur = e;
      return;
    }
  }
  _entries.push_back(e);
}

FunctionConstants&
FunctionConstants::set_bool(unsigned index, bool v)
{
  put_({index, DT::Bool, v ? 1u : 0u});
  return *this;
}

FunctionConstants&
FunctionConstants::set_int(unsigned index, std::int32_t v)
{
  std::uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  put_({index, DT::Int, bits});
  return *this;
}

FunctionConstants&
FunctionConstants::set_uint(unsigned index, std::uint32_t v)
{
  put_({index, DT::UInt, v});
  return *this;
}

FunctionConstants&
FunctionConstants::set_float(unsigned index, float v)
{
  std::uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  put_({index, DT::Float, bits});
  return *this;
}

const std::string&
FunctionConstants::signature() const
{
  if (!_signature_cache.empty() || _entries.empty()) {
    return _signature_cache;
  }
  // Sort by index for a stable representation (set_* may have
  // populated entries in any order).
  std::vector<Entry> sorted(_entries);
  std::sort(sorted.begin(), sorted.end(),
            [](const Entry& a, const Entry& b) {
              return a.index < b.index;
            });
  // Format: 16 bytes per entry, packed: u32 index, u8 dtype,
  // 3 bytes pad, u32 value. Char-stuffed into the string.
  _signature_cache.reserve(sorted.size() * 12);
  for (const Entry& e : sorted) {
    char buf[12];
    std::memcpy(buf + 0,  &e.index, 4);
    buf[4] = static_cast<char>(e.dtype);
    buf[5] = 0;
    buf[6] = 0;
    buf[7] = 0;
    std::memcpy(buf + 8,  &e.value, 4);
    _signature_cache.append(buf, sizeof(buf));
  }
  return _signature_cache;
}

MTL::FunctionConstantValues*
FunctionConstants::build_mtl_() const
{
  MTL::FunctionConstantValues* out =
      MTL::FunctionConstantValues::alloc()->init();
  for (const Entry& e : _entries) {
    MTL::DataType dt = MTL::DataTypeNone;
    switch (e.dtype) {
      case DT::Bool:  dt = MTL::DataTypeBool;  break;
      case DT::Int:   dt = MTL::DataTypeInt;   break;
      case DT::UInt:  dt = MTL::DataTypeUInt;  break;
      case DT::Float: dt = MTL::DataTypeFloat; break;
    }
    // setConstantValue(value, type, index) takes a pointer to the
    // bytes; the value field is the bit pattern of bool/int/uint/
    // float -- all 4 bytes, all 4-byte aligned within Entry.
    out->setConstantValue(&e.value, dt,
                          static_cast<NS::UInteger>(e.index));
  }
  return out;
}

// ---- ComputeFunction --------------------------------------------

ComputeFunction::ComputeFunction(MTL::Function* fn,
                                 MTL::ComputePipelineState* pso) noexcept
  : _fn(fn),
    _pso(pso)
{
}

ComputeFunction::ComputeFunction(ComputeFunction&& o) noexcept
  : _fn(std::exchange(o._fn, nullptr)),
    _pso(std::exchange(o._pso, nullptr))
{
}

ComputeFunction&
ComputeFunction::operator=(ComputeFunction&& o) noexcept
{
  if (this == &o) {
    return *this;
  }
  if (_pso != nullptr) {
    _pso->release();
  }
  if (_fn != nullptr) {
    _fn->release();
  }
  _fn  = std::exchange(o._fn, nullptr);
  _pso = std::exchange(o._pso, nullptr);
  return *this;
}

ComputeFunction::~ComputeFunction()
{
  if (_pso != nullptr) {
    _pso->release();
    _pso = nullptr;
  }
  if (_fn != nullptr) {
    _fn->release();
    _fn = nullptr;
  }
}

unsigned
ComputeFunction::max_total_threads_per_threadgroup() const noexcept
{
  if (_pso == nullptr) {
    return 0;
  }
  return static_cast<unsigned>(_pso->maxTotalThreadsPerThreadgroup());
}

unsigned
ComputeFunction::thread_execution_width() const noexcept
{
  if (_pso == nullptr) {
    return 0;
  }
  return static_cast<unsigned>(_pso->threadExecutionWidth());
}

unsigned
ComputeFunction::static_threadgroup_memory_length() const noexcept
{
  if (_pso == nullptr) {
    return 0;
  }
  return static_cast<unsigned>(_pso->staticThreadgroupMemoryLength());
}

// ---- ComputeLibrary ---------------------------------------------

ComputeLibrary::ComputeLibrary(MTL::Library* lib,
                               std::string name,
                               const MetalCompute* mc) noexcept
  : _lib(lib),
    _name(std::move(name)),
    _mc(mc)
{
}

ComputeLibrary::ComputeLibrary(ComputeLibrary&& o) noexcept
  : _lib(std::exchange(o._lib, nullptr)),
    _name(std::move(o._name)),
    _mc(std::exchange(o._mc, nullptr))
{
}

ComputeLibrary&
ComputeLibrary::operator=(ComputeLibrary&& o) noexcept
{
  if (this == &o) {
    return *this;
  }
  if (_lib != nullptr) {
    _lib->release();
  }
  _lib  = std::exchange(o._lib, nullptr);
  _name = std::move(o._name);
  _mc   = std::exchange(o._mc, nullptr);
  return *this;
}

ComputeLibrary::~ComputeLibrary()
{
  if (_lib != nullptr) {
    _lib->release();
    _lib = nullptr;
  }
}

ComputeFunction
ComputeLibrary::function(std::string_view kernel_name) const
{
  if (_lib == nullptr || _mc == nullptr) {
    return ComputeFunction{};
  }
  return _mc->function(*this, kernel_name);
}

ComputeFunction
ComputeLibrary::function(std::string_view kernel_name,
                         const FunctionConstants& constants) const
{
  if (_lib == nullptr || _mc == nullptr) {
    return ComputeFunction{};
  }
  return _mc->function(*this, kernel_name, constants);
}

}  // namespace vpipe::metal_compute
