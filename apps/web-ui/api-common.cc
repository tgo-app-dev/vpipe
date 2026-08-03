#include "apps/web-ui/api-common.h"

#include <exception>

using namespace std;

namespace vpipe::webui {

bool
blank(const string& s)
{
  for (char c : s) {
    if (c != ' ' && c != '\t' && c != '\n' && c != '\r') { return false; }
  }
  return true;
}

// Strip leading/trailing ASCII whitespace (for user-supplied ids).
string
trim(const string& s)
{
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == string::npos) { return {}; }
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

optional<FlexData>
parse_json_body(const HttpRequest& req)
{
  if (blank(req.body)) { return FlexData::make_object(); }
  try {
    return FlexData::from_json(req.body);
  } catch (const exception&) {
    return nullopt;
  }
}

FlexData
fstr(string_view s)
{
  return FlexData::make_string(s);
}

// Extract a single query-string parameter value (no URL-decoding
// beyond '+'); returns empty when absent. `query` is the raw string
// after '?', e.g. "since=42&foo=bar".
string
query_param(const string& query, const string& key)
{
  size_t i = 0;
  while (i < query.size()) {
    size_t amp = query.find('&', i);
    size_t end = (amp == string::npos) ? query.size() : amp;
    size_t eq  = query.find('=', i);
    if (eq != string::npos && eq < end) {
      if (query.compare(i, eq - i, key) == 0) {
        return query.substr(eq + 1, end - eq - 1);
      }
    }
    if (amp == string::npos) { break; }
    i = amp + 1;
  }
  return {};
}

// Percent-decode a query-string value (e.g. "%2Fa%20b" -> "/a b"). The
// client sends paths via encodeURIComponent, which escapes '/', spaces
// and non-ASCII, so a path query param must be decoded before use.
// '+' is left literal (encodeURIComponent uses %20 for space, %2B for a
// real '+'); a malformed trailing '%' is passed through unchanged.
string
url_decode(const string& s)
{
  auto hex = [](char c) -> int {
    if (c >= '0' && c <= '9') { return c - '0'; }
    if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
    if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
    return -1;
  };
  string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '%' && i + 2 < s.size()) {
      const int hi = hex(s[i + 1]);
      const int lo = hex(s[i + 2]);
      if (hi >= 0 && lo >= 0) {
        out.push_back(static_cast<char>((hi << 4) | lo));
        i += 2;
        continue;
      }
    }
    out.push_back(s[i]);
  }
  return out;
}
}
