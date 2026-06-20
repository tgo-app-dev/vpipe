#ifndef LMDB_TXN_H
#define LMDB_TXN_H

#include "common/session-member.h"
#include <lmdb.h>

namespace vpipe {

class SessionContextIntf;
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
class LmdbTxn : public SessionMember {
public:
  enum class Mode { ReadWrite, ReadOnly };

  LmdbTxn(LmdbEnv& env, Mode mode = Mode::ReadWrite);

  // Explicit-context overload. Uses `ctx` for error routing instead
  // of `env.session()`. Used by code that owns the env *transitively*
  // through the session (so env.session() == Session) but needs
  // errors to bypass the session's delegate -- notably DbLogDelegate,
  // whose error path would otherwise recurse back through itself.
  LmdbTxn(LmdbEnv& env, Mode mode, const SessionContextIntf* ctx);

  LmdbTxn(const LmdbTxn&)            = delete;
  LmdbTxn& operator=(const LmdbTxn&) = delete;
  LmdbTxn(LmdbTxn&&) noexcept;
  LmdbTxn& operator=(LmdbTxn&&) noexcept;
  ~LmdbTxn() override;

  void     commit();          // throws on error
  void     abort() noexcept;  // idempotent
  bool     active() const noexcept;
  Mode     mode()   const noexcept;
  MDB_txn* raw()    const noexcept;

private:
  MDB_txn* _txn;
  Mode     _mode;
  bool     _done;
};

}

#endif
