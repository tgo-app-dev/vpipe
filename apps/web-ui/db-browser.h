#ifndef WEBUI_DB_BROWSER_H
#define WEBUI_DB_BROWSER_H

#include "common/flex-data.h"

#include <string>

namespace vpipe {
class SessionContextIntf;
}

namespace vpipe::webui {

// Read-only browser over the session's LMDB environment, backing the
// Database view. Uses raw LMDB read-only transactions (so a query never
// blocks writers and never throws through the HTTP handler). All three
// calls return a FlexData document to serialize as the JSON response;
// on failure they set `err` (non-empty) and return an empty object.
//
// Key interpretation modes ("text" | "number" | "time" | "auto"):
//   text   -- raw bytes compared/lexicographically ordered as a string
//   number -- key parsed as a number (ASCII decimal, or big-endian
//             binary for 1/2/4/8-byte keys) and compared numerically
//   time   -- key parsed as an instant (epoch number with ms/us/ns
//             auto-scaling, or an ISO-ish date string) and compared
//   auto   -- detected from the database's first key
// Match is "exact" or "range" (lo/hi, either bound optional). An empty
// filter matches every key (browse).
class DbBrowser {
public:
  explicit DbBrowser(SessionContextIntf* sctx) : _sctx(sctx) {}

  // { databases: [ { name, count } ] }
  FlexData list_databases(std::string& err);

  // req: { db, mode, match, q, lo, hi, page }
  // -> { db, mode, match, page, page_size, has_prev, has_next,
  //      truncated, keys: [ { key (base64), display } ] }
  FlexData query_keys(const FlexData& req, std::string& err);

  // req: { db, key (base64) }
  // -> { found, size, encoding ("json"|"hex"), text, display }
  FlexData read_value(const FlexData& req, std::string& err);

  // Mutating ops. These open a WRITE transaction, so the caller must
  // ensure no pipeline stage is writing concurrently (the web-ui gates
  // them on "every pipeline stopped"). On failure they set `err` and
  // return an empty object.

  // req: { db, key (base64) } -> { ok, deleted } (deleted=false if the
  // key was already absent).
  FlexData delete_key(const FlexData& req, std::string& err);

  // req: { db } -> { ok }. Drops the named sub-database entirely (it
  // disappears from list_databases), not just empties it.
  FlexData drop_database(const FlexData& req, std::string& err);

  static constexpr int kPageSize = 20;

private:
  SessionContextIntf* _sctx;
};

}

#endif
