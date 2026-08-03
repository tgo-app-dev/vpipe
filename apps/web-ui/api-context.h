// api-context.h -- the state the /api controllers share.
//
// SessionApi used to be one class holding every /api handler plus all
// of their state. It is now a composition root over per-subject
// controllers; this is what those controllers have in common.
//
// The mutex is the part worth reading. It is ONE lock for every
// controller, exactly as SessionApi::_mu was, and that is deliberate
// rather than left over: the subjects are not actually independent.
// h_db_delete_key_ holds it across its check-and-write specifically so
// a concurrent pipeline launch cannot slip in between, and the profiler
// relies on it to serialize reuse of its one temp file. Giving each
// controller its own lock would look tidier and would quietly widen
// what can run concurrently, which is not a change a file-splitting
// refactor gets to make.
//
// DatabaseApi keeps its own second lock (_db_mu) for LMDB's
// dbi-open serialization. Lock order where both are held: `mu` first.

#ifndef WEBUI_API_CONTEXT_H
#define WEBUI_API_CONTEXT_H

#include <mutex>

namespace vpipe {
class SessionIntf;
class SessionContextIntf;
}

namespace vpipe::webui {

class HttpServer;
class WebUiDelegate;
class WebUiLogDelegate;

struct ApiContext {
  SessionIntf*        session = nullptr;
  SessionContextIntf* sctx    = nullptr;   // same object, context facet
  // Null when the session was not given the corresponding delegate;
  // the controllers behind them then register no routes at all, so the
  // paths 404 rather than answering with an empty document.
  WebUiDelegate*      ui      = nullptr;
  WebUiLogDelegate*   log     = nullptr;
  // The server the routes were registered on, so a long-lived handler
  // (the view WebSocket pump) can see a shutdown coming and unwind.
  // Null until register_routes.
  HttpServer*         server  = nullptr;

  // Serializes every controller. See the note above before narrowing.
  std::mutex          mu;
};

}

#endif
