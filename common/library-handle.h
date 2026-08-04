#ifndef LIBRARY_HANDLE_H
#define LIBRARY_HANDLE_H

#include "interfaces/log-sink-intf.h"
#include <string>
#include <string_view>

namespace vpipe {

// A dlopen'd shared library.
//
// Takes a LOG SINK, not a session context: everything it does with the
// session is report a dlopen or symbol failure. That keeps a low-level
// dlopen wrapper constructible from anything that can log, and off the
// dependency path of the subsystems a session context reaches.
class LibraryHandle {
public:
  enum class LoadMode { Required, Optional };

  // On dlopen failure:
  //   - LoadMode::Required -> log->error(...) (throws)
  //   - LoadMode::Optional -> log->warn(...);   valid() == false
  // flags == 0 means RTLD_NOW | RTLD_LOCAL.
  LibraryHandle(const LogSinkIntf* log,
                std::string_view   path,
                LoadMode           mode  = LoadMode::Required,
                int                flags = 0);

  LibraryHandle(const LibraryHandle&)            = delete;
  LibraryHandle& operator=(const LibraryHandle&) = delete;
  LibraryHandle(LibraryHandle&&) noexcept;
  LibraryHandle& operator=(LibraryHandle&&) noexcept;
  virtual ~LibraryHandle();

  std::string_view path()  const noexcept;
  bool             valid() const noexcept;
  explicit operator bool() const noexcept { return valid(); }

  // Returns nullptr if the symbol is absent.
  void* get_symbol(std::string_view name) const noexcept;

  // Calls log->error(...) on miss (throws).
  void* require_symbol(std::string_view name) const;

  template <typename T>
  T* symbol(std::string_view name) const noexcept {
    return reinterpret_cast<T*>(get_symbol(name));
  }
  template <typename T>
  T* require_symbol_as(std::string_view name) const {
    return reinterpret_cast<T*>(require_symbol(name));
  }

protected:
  // For subclasses that report on their own behalf (the FFmpeg wrappers).
  const LogSinkIntf* log() const noexcept { return _log; }

private:
  const LogSinkIntf* _log;
  void*              _handle;
  std::string        _path;
};

}

#endif
