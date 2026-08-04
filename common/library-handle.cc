#include "common/library-handle.h"
#include "common/vpipe-format.h"
#include "interfaces/log-sink-intf.h"
#include <dlfcn.h>

using namespace std;

namespace vpipe {

LibraryHandle::LibraryHandle(const LogSinkIntf* log,
                             string_view        path,
                             LoadMode           mode,
                             int                flags)
  : _log(log)
  , _handle(nullptr)
  , _path(path)
{
  int eff_flags = flags ? flags : (RTLD_NOW | RTLD_LOCAL);
  _handle = ::dlopen(_path.c_str(), eff_flags);
  if (_handle) {
    return;
  }
  const char* err = ::dlerror();
  string err_msg = err ? err : "unknown error";
  if (mode == LoadMode::Required) {
    _log->error(
      fmt("dlopen failed for {}: {}", _path, err_msg));
    return;
  }
  _log->warn(
    fmt("dlopen failed for {}: {}", _path, err_msg));
}

LibraryHandle::LibraryHandle(LibraryHandle&& other) noexcept
  : _log(other._log)
  , _handle(other._handle)
  , _path(std::move(other._path))
{
  other._handle = nullptr;
}

LibraryHandle&
LibraryHandle::operator=(LibraryHandle&& other) noexcept
{
  if (this == &other) {
    return *this;
  }
  if (_handle) {
    ::dlclose(_handle);
  }
  _handle = other._handle;
  _path   = std::move(other._path);
  other._handle = nullptr;
  return *this;
}

LibraryHandle::~LibraryHandle()
{
  if (_handle) {
    ::dlclose(_handle);
  }
}

string_view
LibraryHandle::path() const noexcept
{
  return _path;
}

bool
LibraryHandle::valid() const noexcept
{
  return _handle != nullptr;
}

void*
LibraryHandle::get_symbol(string_view name) const noexcept
{
  if (!_handle) {
    return nullptr;
  }
  string name_c(name);
  return ::dlsym(_handle, name_c.c_str());
}

void*
LibraryHandle::require_symbol(string_view name) const
{
  void* sym = get_symbol(name);
  if (sym) {
    return sym;
  }
  _log->error(
    fmt("symbol '{}' not found in {}", name, _path));
  return nullptr;
}

}
