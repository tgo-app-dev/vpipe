#ifndef LMDB_DB_H
#define LMDB_DB_H

#include "common/session-member.h"
#include <lmdb.h>
#include <optional>
#include <string>
#include <string_view>

namespace vpipe {

class SessionContextIntf;
class LmdbEnv;
class LmdbTxn;

// Handle to a named sub-database. Opens the dbi inside a brief internal
// write txn at construction. The dbi remains valid for the env's lifetime;
// we deliberately do NOT call mdb_dbi_close (per LMDB docs, that's almost
// always wrong - env close cleans up).
//
// Thread-safety: construction must complete before workers begin using the
// env (two concurrent constructions would queue at LMDB's writer mutex,
// which is correct but easy to mis-time). After that, an LmdbDb may be
// shared freely; per-call thread-safety follows the LmdbTxn passed in.
class LmdbDb : public SessionMember {
public:
  LmdbDb(LmdbEnv& env, std::string_view name, bool create_if_missing = true);

  // Explicit-context overload. Uses `ctx` for runtime error routing
  // instead of `env.session()` -- see LmdbTxn for the same pattern
  // and DbLogDelegate for the use case (recursion firewall).
  LmdbDb(LmdbEnv&                  env,
         const SessionContextIntf* ctx,
         std::string_view          name,
         bool                      create_if_missing = true);

  LmdbDb(const LmdbDb&)            = delete;
  LmdbDb& operator=(const LmdbDb&) = delete;
  LmdbDb(LmdbDb&&) noexcept;
  LmdbDb& operator=(LmdbDb&&) noexcept;
  ~LmdbDb() override                  = default;

  // get: returns nullopt on MDB_NOTFOUND. The returned view points into
  //      LMDB-managed memory and is only valid until the next op on `txn`
  //      or until the txn ends.
  std::optional<std::string_view>
        get(LmdbTxn& txn, std::string_view key) const;

  // put: throws on error. flags forwarded to mdb_put (e.g. MDB_NOOVERWRITE,
  //      MDB_APPEND for monotonic timestamp keys).
  void  put(LmdbTxn&         txn,
            std::string_view key,
            std::string_view value,
            unsigned         flags = 0);

  // del: returns true if a record was removed, false on MDB_NOTFOUND.
  bool  del(LmdbTxn& txn, std::string_view key);

  MDB_dbi          dbi()  const noexcept;
  std::string_view name() const noexcept;

private:
  MDB_dbi     _dbi;
  std::string _name;
};

}

#endif
