#ifndef VPIPE_STAGES_REST_CLIENT_STAGE_H
#define VPIPE_STAGES_REST_CLIENT_STAGE_H

#include "common/flex-data.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpipe {

// REST client stage.
//
// One input (iport0) carries a FlexData payload per beat. On each
// beat the stage issues an HTTP request to the configured endpoint,
// optionally using a sub-tree of the iport payload as the request
// body, and forwards the response (status, headers, body) on oport0
// as a FlexData object.
//
// Body source selection (`payload_path`):
//   The path is a slash-separated JSON-pointer-like expression that
//   walks into the iport0 FlexData. Numeric steps index arrays;
//   string steps look up object keys. Empty / missing path means
//   "use the iport0 payload verbatim". Examples:
//     ""               -> full payload
//     "body"           -> payload.body
//     "messages/0/raw" -> payload.messages[0].raw
//
// Body encoding (`payload_format`):
//   "json"       (default) the selected sub-tree is serialised to
//                JSON; the default Content-Type is application/json.
//   "raw_string" the selected sub-tree must be a string and is sent
//                verbatim; the default Content-Type is text/plain.
//   "none"       no request body is sent (typical for GET/DELETE);
//                Content-Type is omitted unless explicitly set.
//
// Response shape on oport0 (always a FlexData object):
//   {
//     "ok":         <bool>      true iff 2xx
//     "status":     <int>       HTTP status code (0 on transport error)
//     "headers":    <object>    lower-cased header name -> value
//     "body":       <FlexData>  parsed JSON when the response's
//                               Content-Type is application/json,
//                               otherwise the raw body as a string
//     "body_raw":   <string>    raw body bytes (always present)
//     "error":      <string>    transport-level error message; empty
//                               on success
//     "elapsed_ms": <uint>      wall-clock duration of the call
//   }
//
// Configuration (FlexData object):
//   method            (string, required)  HTTP verb (GET/POST/PUT/...).
//   url               (string, required)  full endpoint URL.
//   headers           (object, optional)  name -> value, both strings.
//   payload_path      (string, optional)  selector into iport0 payload.
//   payload_format    (string, optional)  "json" | "raw_string" | "none".
//                                         Default "json".
//   content_type      (string, optional)  overrides the default
//                                         Content-Type for the body.
//   timeout_seconds   (uint,   optional)  network timeout. Default 30.
//   verify_tls        (bool,   optional)  enforce TLS cert validation.
//                                         Default true.
//   emit_on_error     (bool,   optional)  when false, transport errors
//                                         are logged and no beat is
//                                         written on oport0. Default
//                                         true (emit error envelopes).
class RestClientStage final : public TypedStage<RestClientStage> {
public:
  static constexpr const char* kTypeName = "rest-client";

  RestClientStage(const SessionContextIntf* session,
                  std::string               id,
                  std::vector<InEdge>       iports,
                  FlexData                  config);
  ~RestClientStage() override;

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  uint64_t invocations() const noexcept
  {
    return _invocations.load(std::memory_order_acquire);
  }
  int last_status() const noexcept
  {
    return _last_status.load(std::memory_order_acquire);
  }

  // Walk `payload_path` into `root`. Returns true on success and
  // copies the located sub-tree into `out`. Empty path returns the
  // root verbatim. On a missing key, out-of-range index, or wrong
  // kind, returns false and fills `err`. Exposed publicly so the
  // test suite can drive it without a Pipeline.
  static bool
  resolve_path(const FlexData& root,
               const std::string& path,
               FlexData& out,
               std::string& err);

private:
  enum class BodyFormat : unsigned char { Json, RawString, None };

  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*. Declarations carry no non-zero default.
  std::string                                  _method;
  std::string                                  _url;
  std::unordered_map<std::string, std::string> _headers;
  std::string                                  _payload_path;
  BodyFormat                                   _body_format{};
  std::string                                  _content_type_override;
  unsigned                                     _timeout_seconds{};
  bool                                         _verify_tls{};
  bool                                         _emit_on_error{};

  std::atomic<uint64_t> _invocations{0};
  std::atomic<int>      _last_status{0};
};

}

#endif
