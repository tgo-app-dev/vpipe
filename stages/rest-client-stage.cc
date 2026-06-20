#include "stages/rest-client-stage.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <curl/curl.h>

using namespace std;

namespace vpipe {

namespace {

// One-shot global curl initialiser. Called by the stage ctor; safe
// to call from many threads -- curl_global_init is idempotent under
// the documented "may be called more than once" rule, but we still
// gate it with a std::once_flag to keep the call count down.
void
ensure_curl_global_init()
{
  static std::once_flag once;
  std::call_once(once, []{
    curl_global_init(CURL_GLOBAL_DEFAULT);
  });
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

// Try to JSON-parse `body`; on failure return a FlexData string.
FlexData
parse_body_or_string_(string_view body, string_view content_type)
{
  // Treat application/json + */*+json (e.g. application/ld+json,
  // application/vnd.api+json) as JSON. Lower-cased ct expected.
  bool looks_json = false;
  auto semi = content_type.find(';');
  string_view ct = semi == string_view::npos
      ? content_type : content_type.substr(0, semi);
  while (!ct.empty() && std::isspace(static_cast<unsigned char>(ct.back()))) {
    ct.remove_suffix(1);
  }
  if (ct == "application/json"
      || (ct.size() > 5 && ct.compare(ct.size() - 5, 5, "+json") == 0
          && ct.find('/') != string_view::npos)) {
    looks_json = true;
  }
  if (looks_json && !body.empty()) {
    try {
      return FlexData::from_json(body);
    } catch (const std::exception&) {
      // fall through to string
    }
  }
  return FlexData::make_string(body);
}

// libcurl write callback -- appends to a std::string passed via userp.
size_t
curl_write_cb_(char* ptr, size_t size, size_t nmemb, void* userp)
{
  const size_t bytes = size * nmemb;
  static_cast<string*>(userp)->append(ptr, bytes);
  return bytes;
}

// libcurl header callback -- collects lower-cased headers into a
// FlexData object passed via userp. Skips the HTTP status line and
// empty/end-of-headers lines.
size_t
curl_header_cb_(char* buf, size_t size, size_t nitems, void* userp)
{
  const size_t bytes = size * nitems;
  string_view line(buf, bytes);
  // Trim CR/LF on the right.
  while (!line.empty()
         && (line.back() == '\r' || line.back() == '\n')) {
    line.remove_suffix(1);
  }
  if (line.empty()) {
    return bytes;
  }
  // "HTTP/1.1 200 OK" status line has no colon -- skip it.
  auto colon = line.find(':');
  if (colon == string_view::npos) {
    return bytes;
  }
  string name = lower_(line.substr(0, colon));
  string_view value = line.substr(colon + 1);
  while (!value.empty()
         && std::isspace(static_cast<unsigned char>(value.front()))) {
    value.remove_prefix(1);
  }
  while (!value.empty()
         && std::isspace(static_cast<unsigned char>(value.back()))) {
    value.remove_suffix(1);
  }
  auto* obj = static_cast<FlexData*>(userp);
  obj->as_object().insert_or_assign(name, FlexData::make_string(value));
  return bytes;
}

}  // namespace

RestClientStage::RestClientStage(const SessionContextIntf* s,
                                 string                    id,
                                 vector<InEdge>            iports,
                                 FlexData                  config)
  : TypedStage<RestClientStage>(s, std::move(id), std::move(iports),
                                std::move(config))
{
  ensure_curl_global_init();

  // Attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default (and tolerates a non-object
  // config -- handled by the dedicated check below).
  const FlexData& cfg = this->config();
  if (!cfg.is_object()) {
    fail_config(fmt(
        "RestClientStage('{}'): config must be an object", this->id()));
  }

  _method = attr_str("method");
  // Normalise to uppercase so libcurl + comparisons stay simple.
  std::transform(_method.begin(), _method.end(), _method.begin(),
      [](unsigned char c){ return std::toupper(c); });
  if (_method.empty()) {
    fail_config(fmt(
        "RestClientStage('{}'): config.method is required "
        "(GET/POST/PUT/PATCH/DELETE/HEAD)", this->id()));
  } else if (_method != "GET" && _method != "POST"
          && _method != "PUT" && _method != "PATCH"
          && _method != "DELETE" && _method != "HEAD") {
    fail_config(fmt(
        "RestClientStage('{}'): unsupported method '{}'",
        this->id(), _method));
  }

  _url = attr_str("url");
  if (_url.empty()) {
    fail_config(fmt(
        "RestClientStage('{}'): config.url is required",
        this->id()));
  }

  // Composite attribute: empty object when unset (attr() default).
  FlexData h = attr("headers");
  if (!h.is_object()) {
    fail_config(fmt(
        "RestClientStage('{}'): config.headers must be an object",
        this->id()));
  } else {
    for (auto entry : h.as_object()) {
      if (!entry.second.is_string()) {
        fail_config(fmt(
            "RestClientStage('{}'): header '{}' value must be a string",
            this->id(), entry.first));
        break;
      }
      _headers.emplace(string(entry.first),
                       string(entry.second.get_string()));
    }
  }

  _payload_path = attr_str("payload_path");

  string pf = attr_str("payload_format");
  if (pf == "json" || pf.empty()) {
    _body_format = BodyFormat::Json;
  } else if (pf == "raw_string") {
    _body_format = BodyFormat::RawString;
  } else if (pf == "none") {
    _body_format = BodyFormat::None;
  } else {
    fail_config(fmt(
        "RestClientStage('{}'): payload_format must be "
        "'json' | 'raw_string' | 'none' (got '{}')",
        this->id(), pf));
  }

  _content_type_override = attr_str("content_type");

  uint64_t t = attr_uint("timeout_seconds");
  if (t == 0 || t > 3600) {
    fail_config(fmt(
        "RestClientStage('{}'): timeout_seconds must be in [1, 3600]",
        this->id()));
  } else {
    _timeout_seconds = static_cast<unsigned>(t);
  }

  _verify_tls    = attr_bool("verify_tls");
  _emit_on_error = attr_bool("emit_on_error");

  allocate_oports(spec().oports.size());
}

RestClientStage::~RestClientStage() = default;

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "method", .type = ConfigType::String, .required = true,
   .doc = "HTTP method: GET|POST|PUT|PATCH|DELETE|HEAD"},
  {.key = "url", .type = ConfigType::String, .required = true,
   .doc = "full endpoint URL (http/https)"},
  {.key = "headers", .type = ConfigType::Object,
   .doc = "object of header_name -> string value"},
  {.key = "payload_path", .type = ConfigType::String,
   .doc = "slash-separated selector into iport0 payload "
          "(empty = use payload verbatim)"},
  {.key = "payload_format", .type = ConfigType::String,
   .doc = "json | raw_string | none", .def_str = "json"},
  {.key = "content_type", .type = ConfigType::String,
   .doc = "explicit Content-Type override"},
  {.key = "timeout_seconds", .type = ConfigType::Uint,
   .doc = "network timeout (1..3600)", .def_uint = 30},
  {.key = "verify_tls", .type = ConfigType::Bool,
   .doc = "enforce TLS certificate validation", .def_bool = true},
  {.key = "emit_on_error", .type = ConfigType::Bool,
   .doc = "emit a response beat even on transport error",
   .def_bool = true},
};
// iport0 tolerates any beat (FlexDataPayload body, or a TriggerPayload
// driving a bodyless GET) -> untyped. oport0 always emits a FlexData
// response envelope.
const PortSpec kIports[] = {
  {.name = "request", .doc = "FlexData body source, or any trigger beat",
   .type = nullptr, .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "response",
   .doc = "FlexData {ok,status,headers,body,body_raw,error,elapsed_ms}",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "rest-client",
  .doc       = "Issues an HTTP request per input beat (libcurl) and "
               "forwards the response (status/headers/body) as a "
               "FlexData envelope on oport0.",
  .display_name = "REST Client",
  .category  = StageCategory::Network,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
RestClientStage::spec() const noexcept
{
  return kSpec;
}

bool
RestClientStage::resolve_path(const FlexData& root,
                              const string&   path,
                              FlexData&       out,
                              string&         err)
{
  if (path.empty()) {
    out = root;
    return true;
  }
  FlexData cur = root;
  size_t i = 0;
  // Skip a leading '/'; allows both "a/b" and "/a/b".
  if (path[0] == '/') { i = 1; }
  while (i < path.size()) {
    size_t j = path.find('/', i);
    string step(path, i, (j == string::npos ? path.size() : j) - i);
    i = (j == string::npos ? path.size() : j + 1);
    if (step.empty()) { continue; }
    if (cur.is_object()) {
      auto obj = cur.as_object();
      if (!obj.contains(step)) {
        err = fmt("payload_path: object has no key '{}'", step)();
        return false;
      }
      cur = obj.at(step);
    } else if (cur.is_array()) {
      // Parse step as unsigned index.
      char* endp = nullptr;
      errno = 0;
      unsigned long idx = std::strtoul(step.c_str(), &endp, 10);
      if (errno != 0 || endp == step.c_str() || *endp != '\0') {
        err = fmt("payload_path: '{}' is not a valid array index",
                  step)();
        return false;
      }
      auto arr = cur.as_array();
      if (idx >= arr.size()) {
        err = fmt("payload_path: index {} out of range (size {})",
                  idx, arr.size())();
        return false;
      }
      cur = arr.at(idx);
    } else {
      err = fmt("payload_path: cannot index a scalar with step '{}'",
                step)();
      return false;
    }
  }
  out = std::move(cur);
  return true;
}

Job
RestClientStage::process(RuntimeContext& ctx)
{
  auto t = co_await ctx.read(0);
  if (!t) {
    ctx.signal_done();
    co_return;
  }
  if (ctx.stop_requested()) {
    ctx.signal_done();
    co_return;
  }

  // Pull a FlexData view of the payload. Accept FlexDataPayload
  // explicitly; for any other payload type substitute null so the
  // request can still be issued (e.g. a TriggerPayload driving a
  // bodyless GET).
  FlexData in_data = FlexData::make_null();
  if (const auto* fdp = dynamic_cast<const FlexDataPayload*>(t.get())) {
    in_data = fdp->data;
  }

  // Resolve the body source.
  FlexData selected;
  string   sel_err;
  if (!resolve_path(in_data, _payload_path, selected, sel_err)) {
    session()->warn(fmt(
        "RestClientStage('{}'): {}; skipping beat", this->id(), sel_err));
    co_return;
  }

  string body_bytes;
  string default_content_type;
  switch (_body_format) {
    case BodyFormat::Json:
      // Serialise even when the selected value is a JSON string -- the
      // result is a properly-quoted JSON value, which is what an API
      // expecting application/json wants.
      body_bytes = selected.to_json(false);
      default_content_type = "application/json";
      break;
    case BodyFormat::RawString:
      if (!selected.is_string()) {
        session()->warn(fmt(
            "RestClientStage('{}'): payload_format=raw_string requires "
            "the selected payload to be a string; skipping beat",
            this->id()));
        co_return;
      }
      body_bytes = string(selected.get_string());
      default_content_type = "text/plain; charset=utf-8";
      break;
    case BodyFormat::None:
      // Empty body, no default Content-Type.
      break;
  }

  // Headers: caller-supplied take precedence over the format default.
  std::unordered_map<string, string> final_headers = _headers;
  bool have_ct = false;
  for (const auto& [k, _] : final_headers) {
    if (lower_(k) == "content-type") { have_ct = true; break; }
  }
  if (!have_ct) {
    if (!_content_type_override.empty()) {
      final_headers.emplace("Content-Type", _content_type_override);
    } else if (!default_content_type.empty()
               && _body_format != BodyFormat::None) {
      final_headers.emplace("Content-Type", default_content_type);
    }
  }

  // Build the request.
  CURL* curl = curl_easy_init();
  if (!curl) {
    session()->error(fmt(
        "RestClientStage('{}'): curl_easy_init() failed", this->id()));
    co_return;
  }
  // RAII for headers + curl handle so we can't leak on early exits.
  struct curl_slist* hdrs = nullptr;
  for (const auto& [k, v] : final_headers) {
    string line = k + ": " + v;
    hdrs = curl_slist_append(hdrs, line.c_str());
  }

  curl_easy_setopt(curl, CURLOPT_URL, _url.c_str());
  curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, _method.c_str());
  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT,
                   static_cast<long>(_timeout_seconds));
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT,
                   static_cast<long>(_timeout_seconds));
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, _verify_tls ? 1L : 0L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, _verify_tls ? 2L : 0L);

  if (_method == "HEAD") {
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
  }
  if (_body_format != BodyFormat::None) {
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE,
                     static_cast<long>(body_bytes.size()));
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_bytes.data());
  }
  if (hdrs) {
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
  }

  string body;
  FlexData hdr_obj = FlexData::make_object();
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &curl_write_cb_);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, &curl_header_cb_);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, &hdr_obj);

  auto t0 = chrono::steady_clock::now();
  CURLcode rc = curl_easy_perform(curl);
  auto t1 = chrono::steady_clock::now();
  uint64_t elapsed_ms = static_cast<uint64_t>(
      chrono::duration_cast<chrono::milliseconds>(t1 - t0).count());

  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  string transport_err;
  if (rc != CURLE_OK) {
    transport_err = curl_easy_strerror(rc);
  }
  curl_slist_free_all(hdrs);
  curl_easy_cleanup(curl);

  _last_status.store(static_cast<int>(http_code),
                     std::memory_order_release);
  _invocations.fetch_add(1, std::memory_order_acq_rel);

  if (rc != CURLE_OK) {
    session()->warn(fmt(
        "RestClientStage('{}'): {} {} -- {}",
        this->id(), _method, _url, transport_err));
    if (!_emit_on_error) {
      co_return;
    }
  }

  // Build response envelope.
  string ct_lower;
  {
    auto ho = hdr_obj.as_object();
    if (ho.contains("content-type")) {
      ct_lower = string(ho.at("content-type").as_string(""));
      // already lower-cased by curl_header_cb_
    }
  }
  FlexData parsed_body = parse_body_or_string_(body, ct_lower);

  FlexData env = FlexData::make_object();
  auto eo = env.as_object();
  bool ok = rc == CURLE_OK && http_code >= 200 && http_code < 300;
  eo.insert("ok", FlexData::make_bool(ok));
  eo.insert("status", FlexData::make_int(http_code));
  eo.insert("headers", std::move(hdr_obj));
  eo.insert("body", std::move(parsed_body));
  eo.insert("body_raw", FlexData::make_string(body));
  eo.insert("error", FlexData::make_string(transport_err));
  eo.insert("elapsed_ms", FlexData::make_uint(elapsed_ms));

  co_await ctx.write(0,
      make_payload<FlexDataPayload>(std::move(env)));
}

VPIPE_REGISTER_STAGE(RestClientStage)
VPIPE_REGISTER_SPEC(RestClientStage, kSpec)

}
