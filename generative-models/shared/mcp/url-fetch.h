#ifndef VPIPE_GENERATIVE_MODELS_URL_FETCH_H
#define VPIPE_GENERATIVE_MODELS_URL_FETCH_H

#include "generative-models/shared/mcp/mcp-tools.h"

#include <cstddef>
#include <string>

namespace vpipe {

// Knobs for one URL fetch. Defaults suit a chat tool: a modest body cap
// and short timeout, SSRF-blocked (allow_private=false).
struct UrlFetchOptions {
  int         timeout_seconds = 15;
  std::size_t max_bytes       = 1024 * 1024;   // downloaded-body cap
  std::size_t max_text_chars  = 16 * 1024;     // scrape_page text cap
  int         max_redirects   = 5;
  // When false (the safe default), requests to loopback / private /
  // link-local / CGNAT / multicast addresses -- and the localhost name
  // -- are refused (SSRF guard). Set true only for trusted local use.
  bool        allow_private   = false;
};

struct UrlFetchResult {
  bool        ok = false;          // request completed (any HTTP status)
  long        status = 0;          // HTTP response code
  std::string content_type;
  std::string final_url;           // after redirects
  std::string body;                // capped at max_bytes
  bool        truncated = false;   // body hit the cap
  std::string error;               // set when ok == false
};

// GET `url` over http/https with TLS verification, a size cap, a redirect
// cap, a timeout, and SSRF protection (see UrlFetchOptions::allow_private).
// Returns ok=false + error on a blocked target, a non-http(s) scheme, or a
// transport failure.
UrlFetchResult
fetch_url(const std::string& url, const UrlFetchOptions& opts = {});

// Reduce an HTML document to readable plain text: drop <script>/<style>
// blocks, turn block-level tags into line breaks, strip the remaining
// tags, decode common entities, collapse whitespace, and cap the result
// at `max_chars`. Non-HTML text passes through (tags absent).
std::string
html_to_text(const std::string& html, std::size_t max_chars);

// Extract the <title> text from an HTML document, or "" when absent.
std::string
html_title(const std::string& html);

// The MCP tools. fetch_url returns the raw (text) body; scrape_page
// returns the readable text of a web page. Both carry `opts` (so the
// SSRF policy / caps are fixed at registration).
McpTool make_fetch_url_tool(const UrlFetchOptions& opts = {});
McpTool make_scrape_page_tool(const UrlFetchOptions& opts = {});

}

#endif
