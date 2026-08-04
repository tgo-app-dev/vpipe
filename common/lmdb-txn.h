#ifndef LMDB_TXN_H
#define LMDB_TXN_H

#include "interfaces/log-sink-intf.h"
#include <lmdb.h>

namespace vpipe {

class LmdbEnv;

// RAII over MDB_txn*. Defaults to abort on destruction; an explicit
// commit() leaves the dtor a no-op.
//
// Thread-safety: a txn instance must not be used by two threads at once.
// LmdbEnv is opened with MDB_NOTLS so a read-only txn may migrate between
// threads as long as the application serializes access (e.g. a job carrying
// the txn isn't stolen mid-flight). Write txns serialize globally per env:
// a second mdb_txn_begin(ReadWrite) blocks inside LMDB until the prior
// writer commits or aborts.
class LmdbTxn {
public:
  enum class Mode { ReadWrite, ReadOnly };

  LmdbTxn(LmdbEnv& env, Mode mode = Mode::ReadWrite);

  // Explicit-context overload. Uses `ctx` for error routing instead
  // of `env.log()`. Used by code that owns the env *transitively*
  // through the session (so env.log() is that session) but needs
  // errors to bypass the session's delegate -- notably DbLogDelegate,
  // whose error path would otherwise recurse back through itself.
  LmdbTxn(LmdbEnv& env, Mode mode, const LogSinkIntf* ctx);

  LmdbTxn(const LmdbTxn&)            = delete;
  LmdbTxn& operator=(const LmdbTxn&) = delete;
  LmdbTxn(LmdbTxn&&) noexcept;
  LmdbTxn& operator=(LmdbTxn&&) noexcept;
  ~LmdbTxn();

  void     commit();          // throws on error
  void     abort() noexcept;  // idempotent
  bool     active() const noexcept;
  Mode     mode()   const noexcept;
  MDB_txn* raw()    const noexcept;

  // The sink these report through. Public because the sibling LMDB
  // types chain off it (LmdbTxn(env), LmdbDb(env), LmdbCursor(txn))
  // exactly where they used to chain off env.session().
  const LogSinkIntf* log() const noexcept { return _log; }

private:
  const LogSinkIntf* _log = nullptr;

  MDB_txn* _txn;
  Mode     _mode;
  bool     _done;
};

}

#endif
