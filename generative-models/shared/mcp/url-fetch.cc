#include "generative-models/shared/mcp/url-fetch.h"

#include "common/flex-data.h"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <curl/curl.h>

using namespace std;

namespace vpipe {

namespace {

void
ensure_curl_global_init_()
{
  static std::once_flag once;
  std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

string
lower_(string_view s)
{
  string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back(static_cast<char>(
        std::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

// True when `ip` (a numeric address literal) is loopback / private /
// link-local / CGNAT / multicast / reserved -- the set an SSRF guard
// must refuse. Non-literals (DNS names) return false; the resolved peer
// is checked separately in the prereq callback.
bool
is_blocked_addr_(const char* ip)
{
  if (!ip || !*ip) {
    return false;
  }
  in_addr v4{};
  if (::inet_pton(AF_INET, ip, &v4) == 1) {
    const std::uint32_t a = ntohl(v4.s_addr);
    const std::uint8_t o1 = (a >> 24) & 0xff;
    const std::uint8_t o2 = (a >> 16) & 0xff;
    if (o1 == 0)   return true;                        // 0.0.0.0/8
    if (o1 == 127) return true;                        // loopback
    if (o1 == 10)  return true;                        // private
    if (o1 == 172 && o2 >= 16 && o2 <= 31) return true;// private
    if (o1 == 192 && o2 == 168) return true;           // private
    if (o1 == 169 && o2 == 254) return true;           // link-local + meta
    if (o1 == 100 && o2 >= 64 && o2 <= 127) return true;// CGNAT
    if (o1 >= 224) return true;                        // multicast/reserved
    return false;
  }
  in6_addr v6{};
  if (::inet_pton(AF_INET6, ip, &v6) == 1) {
    const std::uint8_t* b = v6.s6_addr;
    bool all_zero = true;
    for (int i = 0; i < 16; ++i) {
      if (b[i]) { all_zero = false; break; }
    }
    if (all_zero) {
      return true;                                     // ::
    }
    bool loopback = true;
    for (int i = 0; i < 15; ++i) {
      if (b[i]) { loopback = false; break; }
    }
    if (loopback && b[15] == 1) {
      return true;                                     // ::1
    }
    if ((b[0] & 0xfe) == 0xfc) return true;            // fc00::/7 ULA
    if (b[0] == 0xfe && (b[1] & 0xc0) == 0x80) return true; // fe80::/10
    if (b[0] == 0xff) return true;                     // multicast
    // ::ffff:a.b.c.d IPv4-mapped -> re-check the embedded v4.
    static const std::uint8_t kMapped[12] =
        { 0,0,0,0,0,0,0,0,0,0,0xff,0xff };
    if (std::memcmp(b, kMapped, 12) == 0) {
      char v4s[16];
      std::snprintf(v4s, sizeof v4s, "%u.%u.%u.%u",
                    b[12], b[13], b[14], b[15]);
      return is_blocked_addr_(v4s);
    }
    return false;
  }
  return false;
}

// Prereq callback context + hook: libcurl calls this after connecting
// (once per connection, so it also covers redirects) with the peer IP.
// Aborting here blocks DNS-rebinding and redirect-to-internal SSRF.
struct PrereqCtx {
  bool   allow_private = false;
  string blocked_ip;
};

int
prereq_cb_(void* clientp, char* primary_ip, char* /*local_ip*/,
           int /*primary_port*/, int /*local_port*/)
{
  auto* c = static_cast<PrereqCtx*>(clientp);
  if (!c->allow_private && is_blocked_addr_(primary_ip)) {
    c->blocked_ip = primary_ip ? primary_ip : "";
    return CURL_PREREQFUNC_ABORT;
  }
  return CURL_PREREQFUNC_OK;
}

// Write sink with a hard byte cap; returning short aborts the transfer
// (CURLE_WRITE_ERROR), which fetch_url treats as a clean truncation.
struct WriteSink {
  string* body = nullptr;
  size_t  cap  = 0;
  bool    truncated = false;
};

size_t
write_cb_(char* ptr, size_t size, size_t nmemb, void* userp)
{
  auto*  s = static_cast<WriteSink*>(userp);
  const size_t n = size * nmemb;
  if (s->body->size() >= s->cap) {
    s->truncated = true;
    return 0;
  }
  const size_t room = s->cap - s->body->size();
  const size_t take = n <= room ? n : room;
  s->body->append(ptr, take);
  if (take < n) {
    s->truncated = true;
    return 0;
  }
  return n;
}

// Early SSRF / scheme check on the URL as written (before any DNS). Fills
// *err and returns false on a bad scheme or a private/localhost literal.
bool
precheck_url_(const string& url, bool allow_private, string* err)
{
  CURLU* h = curl_url();
  if (!h) {
    *err = "internal url parser error";
    return false;
  }
  bool ok = false;
  char* scheme = nullptr;
  char* host   = nullptr;
  do {
    if (curl_url_set(h, CURLUPART_URL, url.c_str(), 0) != CURLUE_OK) {
      *err = "invalid or unsupported URL";
      break;
    }
    curl_url_get(h, CURLUPART_SCHEME, &scheme, 0);
    const string sch = scheme ? lower_(scheme) : string();
    if (sch != "http" && sch != "https") {
      *err = "only http and https URLs are allowed";
      break;
    }
    curl_url_get(h, CURLUPART_HOST, &host, 0);
    string hs = host ? lower_(host) : string();
    if (hs.empty()) {
      *err = "URL has no host";
      break;
    }
    // Some curl builds return an IPv6 host bracketed ("[::1]"); strip the
    // brackets so the literal check parses it.
    if (hs.size() >= 2 && hs.front() == '[' && hs.back() == ']') {
      hs = hs.substr(1, hs.size() - 2);
    }
    if (!allow_private) {
      if (hs == "localhost" || hs.ends_with(".localhost")
          || is_blocked_addr_(hs.c_str())) {
        *err = "blocked: refusing a private / localhost target";
        break;
      }
    }
    ok = true;
  } while (false);
  curl_free(scheme);
  curl_free(host);
  curl_url_cleanup(h);
  return ok;
}

}  // namespace

UrlFetchResult
fetch_url(const string& url, const UrlFetchOptions& opts)
{
  UrlFetchResult r;
  if (!precheck_url_(url, opts.allow_private, &r.error)) {
    return r;
  }
  ensure_curl_global_init_();
  CURL* curl = curl_easy_init();
  if (!curl) {
    r.error = "curl_easy_init failed";
    return r;
  }
  WriteSink sink{ &r.body, opts.max_bytes, false };
  PrereqCtx pctx{ opts.allow_private, {} };

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  // Lock the allowed schemes for the request AND for redirects so a
  // redirect can't drop to file:// / gopher:// / dict:// etc.
  curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "http,https");
  curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS_STR, "http,https");
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_MAXREDIRS,
                   static_cast<long>(opts.max_redirects));
  curl_easy_setopt(curl, CURLOPT_TIMEOUT,
                   static_cast<long>(opts.timeout_seconds));
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT,
                   static_cast<long>(opts.timeout_seconds));
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
  curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");   // auto-decompress
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "vpipe-mcp-fetch/1");
  curl_easy_setopt(curl, CURLOPT_MAXFILESIZE,
                   static_cast<long>(opts.max_bytes));
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &write_cb_);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &sink);
  curl_easy_setopt(curl, CURLOPT_PREREQFUNCTION, &prereq_cb_);
  curl_easy_setopt(curl, CURLOPT_PREREQDATA, &pctx);

  const CURLcode rc = curl_easy_perform(curl);
  const bool clean_trunc = (rc == CURLE_WRITE_ERROR && sink.truncated);
  if (rc == CURLE_OK || clean_trunc) {
    r.ok = true;
    long code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    r.status = code;
    char* ct = nullptr;
    curl_easy_getinfo(curl, CURLINFO_CONTENT_TYPE, &ct);
    if (ct) { r.content_type = ct; }
    char* eff = nullptr;
    curl_easy_getinfo(curl, CURLINFO_EFFECTIVE_URL, &eff);
    r.final_url = eff ? eff : url;
    r.truncated = sink.truncated;
  } else if (!pctx.blocked_ip.empty()
             || rc == CURLE_ABORTED_BY_CALLBACK) {
    r.error = "blocked: target resolved to a private / local address"
        + (pctx.blocked_ip.empty() ? string()
                                   : " (" + pctx.blocked_ip + ")");
  } else {
    r.error = curl_easy_strerror(rc);
  }
  curl_easy_cleanup(curl);
  return r;
}

// ---- HTML -> text -------------------------------------------------------

namespace {

size_t
ci_find_(const string& hay, const char* needle, size_t from)
{
  const string n = lower_(needle);
  const string h = lower_(hay);   // simple; inputs are page-sized
  return h.find(n, from);
}

// Remove every open..close block (case-insensitive), inclusive.
string
remove_blocks_(string s, const char* open, const char* close)
{
  size_t from = 0;
  for (;;) {
    const size_t a = ci_find_(s, open, from);
    if (a == string::npos) {
      break;
    }
    const size_t b = ci_find_(s, close, a + std::strlen(open));
    if (b == string::npos) {
      s.erase(a);                 // unterminated -> drop the tail
      break;
    }
    s.erase(a, (b + std::strlen(close)) - a);
    from = a;
  }
  return s;
}

void
append_utf8_(string& out, long cp)
{
  if (cp <= 0x7f) {
    out.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7ff) {
    out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else if (cp <= 0xffff) {
    out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else {
    out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  }
}

string
decode_entities_(const string& in)
{
  string out;
  out.reserve(in.size());
  for (size_t i = 0; i < in.size();) {
    if (in[i] != '&') {
      out.push_back(in[i++]);
      continue;
    }
    const size_t semi = in.find(';', i + 1);
    if (semi == string::npos || semi - i > 12) {
      out.push_back(in[i++]);
      continue;
    }
    const string ent = in.substr(i + 1, semi - i - 1);
    bool matched = true;
    if (ent == "amp") out.push_back('&');
    else if (ent == "lt") out.push_back('<');
    else if (ent == "gt") out.push_back('>');
    else if (ent == "quot") out.push_back('"');
    else if (ent == "apos") out.push_back('\'');
    else if (ent == "nbsp") out.push_back(' ');
    else if (!ent.empty() && ent[0] == '#') {
      long cp = -1;
      if (ent.size() > 1 && (ent[1] == 'x' || ent[1] == 'X')) {
        cp = std::strtol(ent.c_str() + 2, nullptr, 16);
      } else {
        cp = std::strtol(ent.c_str() + 1, nullptr, 10);
      }
      if (cp > 0 && cp <= 0x10ffff) {
        append_utf8_(out, cp);
      } else {
        matched = false;
      }
    } else {
      matched = false;
    }
    if (matched) {
      i = semi + 1;
    } else {
      out.push_back(in[i++]);
    }
  }
  return out;
}

// Does the tag (raw inner text, no angle brackets) name a block-level
// element whose open/close should become a line break?
bool
tag_breaks_(const string& raw)
{
  size_t k = 0;
  if (k < raw.size() && raw[k] == '/') {
    ++k;
  }
  string name;
  while (k < raw.size()
         && (std::isalnum(static_cast<unsigned char>(raw[k])))) {
    name.push_back(static_cast<char>(
        std::tolower(static_cast<unsigned char>(raw[k]))));
    ++k;
  }
  static const char* kBreak[] = {
    "p","div","br","hr","li","ul","ol","tr","td","th","table",
    "h1","h2","h3","h4","h5","h6","section","article","header",
    "footer","nav","blockquote","pre","figure","figcaption","form"
  };
  for (const char* b : kBreak) {
    if (name == b) {
      return true;
    }
  }
  return false;
}

string
collapse_ws_(const string& in)
{
  // Split into lines, collapse intra-line whitespace + trim, then join
  // with at most one blank line between blocks.
  vector<string> lines;
  string cur;
  for (char c : in) {
    if (c == '\n') {
      lines.push_back(std::move(cur));
      cur.clear();
    } else if (c != '\r') {
      cur.push_back(c);
    }
  }
  lines.push_back(std::move(cur));
  string out;
  int blank = 0;
  for (auto& L : lines) {
    string t;
    bool sp = false;
    for (char c : L) {
      if (c == ' ' || c == '\t') {
        if (!sp) { t.push_back(' '); sp = true; }
      } else {
        t.push_back(c);
        sp = false;
      }
    }
    const size_t b = t.find_first_not_of(' ');
    const size_t e = t.find_last_not_of(' ');
    const string line = (b == string::npos) ? "" : t.substr(b, e - b + 1);
    if (line.empty()) {
      if (++blank <= 1 && !out.empty()) {
        out.push_back('\n');
      }
    } else {
      blank = 0;
      if (!out.empty() && out.back() != '\n') {
        out.push_back('\n');
      }
      out += line;
    }
  }
  const size_t b = out.find_first_not_of("\n");
  const size_t e = out.find_last_not_of("\n");
  return (b == string::npos) ? "" : out.substr(b, e - b + 1);
}

}  // namespace

string
html_to_text(const string& html, size_t max_chars)
{
  string s = remove_blocks_(html, "<script", "</script>");
  s = remove_blocks_(s, "<style", "</style>");
  s = remove_blocks_(s, "<!--", "-->");
  string out;
  out.reserve(s.size());
  bool in_tag = false;
  string tag;
  for (char c : s) {
    if (!in_tag) {
      if (c == '<') { in_tag = true; tag.clear(); }
      else out.push_back(c);
    } else {
      if (c == '>') {
        in_tag = false;
        if (tag_breaks_(tag)) { out.push_back('\n'); }
      } else {
        tag.push_back(c);
      }
    }
  }
  out = decode_entities_(out);
  out = collapse_ws_(out);
  if (out.size() > max_chars) {
    out.resize(max_chars);
  }
  return out;
}

string
html_title(const string& html)
{
  const size_t a = ci_find_(html, "<title", 0);
  if (a == string::npos) {
    return {};
  }
  const size_t gt = html.find('>', a);
  if (gt == string::npos) {
    return {};
  }
  const size_t b = ci_find_(html, "</title>", gt + 1);
  const string inner = html.substr(gt + 1,
      (b == string::npos ? html.size() : b) - (gt + 1));
  return collapse_ws_(decode_entities_(inner));
}

// ---- tools --------------------------------------------------------------

namespace {

bool
is_texty_(const string& content_type)
{
  const string ct = lower_(content_type);
  if (ct.empty()) {
    return true;   // unknown -> assume text, but NUL check still guards
  }
  return ct.find("text/") != string::npos
      || ct.find("json") != string::npos
      || ct.find("xml") != string::npos
      || ct.find("html") != string::npos
      || ct.find("javascript") != string::npos
      || ct.find("csv") != string::npos;
}

string
url_arg_(const string& args_json)
{
  try {
    FlexData a = FlexData::from_json(args_json);
    if (a.is_object()) {
      auto o = a.as_object();
      if (o.contains("url")) {
        FlexData u = o.at("url");
        if (u.is_string()) { return string(u.get_string()); }
      }
    }
  } catch (const std::exception&) {
  }
  return {};
}

FlexData
error_obj_(const string& msg)
{
  FlexData fd = FlexData::make_object();
  fd.as_object().insert("error", FlexData::make_string(msg));
  return fd;
}

}  // namespace

McpTool
make_fetch_url_tool(const UrlFetchOptions& opts)
{
  McpTool t;
  t.name = "fetch_url";
  t.description =
      "Fetch a public http(s) URL and return the response body as text. "
      "Private/localhost addresses are refused. Binary responses are not "
      "returned.";
  t.parameters_json =
      "{\"type\":\"object\",\"properties\":{\"url\":{\"type\":\"string\","
      "\"description\":\"absolute http(s) URL\"}},\"required\":[\"url\"]}";
  t.handler = [opts](const string& args_json) -> string {
    const string url = url_arg_(args_json);
    if (url.empty()) {
      return error_obj_("missing 'url' argument").to_json(false);
    }
    const UrlFetchResult r = fetch_url(url, opts);
    if (!r.ok) {
      return error_obj_(r.error).to_json(false);
    }
    FlexData fd = FlexData::make_object();
    auto o = fd.as_object();
    o.insert("url", FlexData::make_string(r.final_url));
    o.insert("status", FlexData::make_int(r.status));
    o.insert("content_type", FlexData::make_string(r.content_type));
    if (is_texty_(r.content_type)
        && r.body.find('\0') == string::npos) {
      string body = r.body;
      bool clipped = r.truncated;
      if (body.size() > opts.max_text_chars) {
        body.resize(opts.max_text_chars);
        clipped = true;
      }
      o.insert("content", FlexData::make_string(body));
      if (clipped) { o.insert("truncated", FlexData::make_bool(true)); }
    } else {
      o.insert("note",
               FlexData::make_string("non-text content not returned"));
      o.insert("bytes",
               FlexData::make_int(static_cast<int64_t>(r.body.size())));
    }
    return fd.to_json(false);
  };
  return t;
}

McpTool
make_scrape_page_tool(const UrlFetchOptions& opts)
{
  McpTool t;
  t.name = "scrape_page";
  t.description =
      "Fetch a public web page and return its readable text with HTML "
      "markup removed. Private/localhost addresses are refused.";
  t.parameters_json =
      "{\"type\":\"object\",\"properties\":{\"url\":{\"type\":\"string\","
      "\"description\":\"absolute http(s) URL of a web page\"}},"
      "\"required\":[\"url\"]}";
  t.handler = [opts](const string& args_json) -> string {
    const string url = url_arg_(args_json);
    if (url.empty()) {
      return error_obj_("missing 'url' argument").to_json(false);
    }
    const UrlFetchResult r = fetch_url(url, opts);
    if (!r.ok) {
      return error_obj_(r.error).to_json(false);
    }
    const string title = html_title(r.body);
    const string text  = html_to_text(r.body, opts.max_text_chars);
    FlexData fd = FlexData::make_object();
    auto o = fd.as_object();
    o.insert("url", FlexData::make_string(r.final_url));
    o.insert("status", FlexData::make_int(r.status));
    if (!title.empty()) {
      o.insert("title", FlexData::make_string(title));
    }
    o.insert("text", FlexData::make_string(text));
    if (r.truncated) {
      o.insert("truncated", FlexData::make_bool(true));
    }
    return fd.to_json(false);
  };
  return t;
}

}
