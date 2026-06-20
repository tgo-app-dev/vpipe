#ifndef VPIPE_FORMAT_H
#define VPIPE_FORMAT_H

#include <format>
#include <functional>
#include <string>
#include <string_view>

namespace vpipe {

struct VpipeFormat {
  typedef std::function<std::string()> formatter_t;
  formatter_t _f;
  VpipeFormat(formatter_t f) : _f(std::move(f)) {};
  std::string operator()() const { return _f(); };
  VpipeFormat(VpipeFormat&&) = default;
  VpipeFormat(const VpipeFormat&) = default;
};

template <typename... Args>
VpipeFormat
fmt(std::string_view f, Args&&... args) {
  return VpipeFormat([=]()
                     { return std::vformat(f, std::make_format_args(args...));
                     });
}

}

#endif

