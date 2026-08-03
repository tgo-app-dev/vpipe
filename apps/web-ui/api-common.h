// api-common.h -- request/JSON helpers shared by the /api controllers.
//
// SessionApi was one class serving every /api route; it is now a
// composition root over per-subject controllers (PipelineApi, FileApi,
// DatabaseApi, ProfilerApi, IoApi, LogApi, ViewApi, SystemApi). These
// are the few helpers that genuinely cross those boundaries -- every
// controller parses a JSON body, builds a string FlexData, or reads a
// query parameter.
//
// Anything used by ONE controller stays file-static in that
// controller's .cc, where it can be read next to its only caller. The
// bar for landing here is more than one subject actually needing it.

#ifndef WEBUI_API_COMMON_H
#define WEBUI_API_COMMON_H

#include "apps/web-ui/http-server.h"
#include "common/flex-data.h"

#include <optional>
#include <string>
#include <string_view>

namespace vpipe::webui {

// True when `s` is empty or only ASCII whitespace.
bool blank(const std::string& s);

// Strip leading/trailing ASCII whitespace (for user-supplied ids).
std::string trim(const std::string& s);

// Parse a request body as JSON. An empty/whitespace body is an empty
// OBJECT, not an error, so a handler taking no arguments needs no
// special case; nullopt means the body was present but malformed,
// which every caller turns into a 400.
std::optional<FlexData> parse_json_body(const HttpRequest& req);

// FlexData string, spelled short because the response builders use it
// constantly.
FlexData fstr(std::string_view s);

// One query-string parameter value from the raw string after '?'
// (e.g. "since=42&foo=bar"); empty when absent. Does NOT percent-decode
// -- pass the result through url_decode when the value may be a path.
std::string query_param(const std::string& query, const std::string& key);

// Percent-decode a query-string value ("%2Fa%20b" -> "/a b"). The
// client sends paths via encodeURIComponent, which escapes '/', spaces
// and non-ASCII, so a path parameter must be decoded before use. '+' is
// left literal (encodeURIComponent emits %20 for space, %2B for a real
// '+'); a malformed trailing '%' passes through unchanged.
std::string url_decode(const std::string& s);

}

#endif
