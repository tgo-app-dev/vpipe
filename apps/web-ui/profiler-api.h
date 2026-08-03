// profiler-api.h -- /api/profiler/* : the session's per-worker event
// capture, started and stopped from the browser.

#ifndef WEBUI_PROFILER_API_H
#define WEBUI_PROFILER_API_H

#include "apps/web-ui/api-context.h"
#include "apps/web-ui/http-server.h"
#include "common/flex-data.h"

#include <string>

namespace vpipe::webui {

// Performance profiler: start/stop the session's per-worker event
// capture and retrieve the captured timeline. start accepts optional
// {max_events}; data returns the dump_profiling document (live while
// capturing, else the snapshot taken at stop).
class ProfilerApi {
public:
  explicit ProfilerApi(ApiContext& ctx) : _ctx(ctx) {}

  void register_routes(HttpServer& s);

private:
  HttpResponse h_start_ (const HttpRequest&);
  HttpResponse h_stop_  (const HttpRequest&);
  HttpResponse h_reset_ (const HttpRequest&);
  HttpResponse h_status_(const HttpRequest&);
  HttpResponse h_data_  (const HttpRequest&);

  // Snapshot the live profiling buffers to a JSON string. There is no
  // in-memory dump API, so this dumps to a temp file (dump_profiling)
  // and reads it back. Empty + `err` set on failure. Caller holds
  // _ctx.mu.
  std::string dump_json_(std::string& err);
  // {enabled, max_events_per_thread, has_data}. Caller holds _ctx.mu.
  FlexData    status_doc_() const;

  ApiContext& _ctx;
  // Last profiling capture (dump_profiling JSON), retained after stop
  // so the timeline survives disabling (which frees the buffers).
  std::string _snapshot;
};

}

#endif
