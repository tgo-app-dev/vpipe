#ifndef LIBRARY_HANDLE_H
#define LIBRARY_HANDLE_H

#include "common/session-member.h"
#include <string>
#include <string_view>

namespace vpipe {

class SessionContextIntf;

class LibraryHandle : public SessionMember {
public:
  enum class LoadMode { Required, Optional };

  // On dlopen failure:
  //   - LoadMode::Required -> session()->error(...) (throws)
  //   - LoadMode::Optional -> session()->warn(...);  valid() == false
  // flags == 0 means RTLD_NOW | RTLD_LOCAL.
  LibraryHandle(const SessionContextIntf* session,
                std::string_view          path,
                LoadMode                  mode  = LoadMode::Required,
                int                       flags = 0);

  LibraryHandle(const LibraryHandle&)            = delete;
  LibraryHandle& operator=(const LibraryHandle&) = delete;
  LibraryHandle(LibraryHandle&&) noexcept;
  LibraryHandle& operator=(LibraryHandle&&) noexcept;
  ~LibraryHandle() override;

  std::string_view path()  const noexcept;
  bool             valid() const noexcept;
  explicit operator bool() const noexcept { return valid(); }

  // Returns nullptr if the symbol is absent.
  void* get_symbol(std::string_view name) const noexcept;

  // Calls session()->error(...) on miss (throws).
  void* require_symbol(std::string_view name) const;

  template <typename T>
  T* symbol(std::string_view name) const noexcept {
    return reinterpret_cast<T*>(get_symbol(name));
  }
  template <typename T>
  T* require_symbol_as(std::string_view name) const {
    return reinterpret_cast<T*>(require_symbol(name));
  }

private:
  void*       _handle;
  std::string _path;
};

}

#endif
