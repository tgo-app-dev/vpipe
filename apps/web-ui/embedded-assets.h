#ifndef WEBUI_EMBEDDED_ASSETS_H
#define WEBUI_EMBEDDED_ASSETS_H

#include <cstddef>
#include <string_view>

namespace vpipe::webui {

// A web-ui asset embedded into the binary at build time (see
// cmake/embed-web-assets.cmake, which generates the registry from the
// web/ tree). The bytes have static storage duration and outlive the
// process's use of them; `data` is NOT NUL-terminated -- always use
// `size`.
struct EmbeddedAsset {
  const unsigned char* data;
  std::size_t          size;
};

// Look up an embedded asset by its URL path ("/index.html",
// "/js/app.js", ...). Returns nullptr if no asset was embedded for that
// path. The returned pointer is valid for the process lifetime.
const EmbeddedAsset* find_embedded_asset(std::string_view path);

// Number of assets embedded at build time (0 when none were embedded).
std::size_t embedded_asset_count();

}  // namespace vpipe::webui

#endif
