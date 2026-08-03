// database-api.h -- /api/db/* : the LMDB browser.
//
// list/keys/value/scan are read-only; delete-key and drop mutate and
// are refused while any pipeline is non-stopped, since a running stage
// could be writing. That check is why this controller holds a
// reference to PipelineApi rather than tracking state of its own -- and
// why it takes ApiContext::mu across the check AND the write, so a
// launch cannot slip in between.

#ifndef WEBUI_DATABASE_API_H
#define WEBUI_DATABASE_API_H

#include "apps/web-ui/api-context.h"
#include "apps/web-ui/http-server.h"
#include "apps/web-ui/pipeline-api.h"

#include <mutex>

namespace vpipe::webui {

class DatabaseApi {
public:
  DatabaseApi(ApiContext& ctx, PipelineApi& pipelines)
    : _ctx(ctx), _pipelines(pipelines) {}

  void register_routes(HttpServer& s);

private:
  // list reports a `deletable` flag so the view shows its delete
  // controls only when mutation is allowed.
  HttpResponse h_list_(const HttpRequest&);
  HttpResponse h_keys_(const HttpRequest&);
  // Streaming value-filtered scan: writes NDJSON records (meta / row* /
  // done) incrementally as matches are found, so a large result set (up
  // to 64k rows) reaches the client progressively.
  void         h_scan_stream_(const HttpRequest&, ResponseStream&);
  HttpResponse h_value_(const HttpRequest&);
  HttpResponse h_delete_key_(const HttpRequest&);
  HttpResponse h_drop_(const HttpRequest&);

  // GET /api/models/installed -- the registered models, enriched with
  // catalogue metadata for the compatibility-aware model browser. Not a
  // /api/db route, but it reads the registry through LMDB and so needs
  // the same dbi-open serialization; keeping it here is cheaper than
  // publishing _db_mu.
  HttpResponse h_models_installed_(const HttpRequest&);

  ApiContext&  _ctx;
  PipelineApi& _pipelines;
  // Serialises the DB handlers: the HTTP server serves requests on
  // per-connection threads, and LMDB's mdb_dbi_open must not run from
  // concurrent transactions. Lock order where both are held
  // (delete/drop): _ctx.mu before _db_mu.
  std::mutex   _db_mu;
};

}

#endif
