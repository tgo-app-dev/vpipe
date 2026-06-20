#include "common/lmdb-txn.h"
#include "common/lmdb-env.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

namespace vpipe {

LmdbTxn::LmdbTxn(LmdbEnv& env, Mode mode)
  : LmdbTxn(env, mode, env.session())
{
}

LmdbTxn::LmdbTxn(LmdbEnv& env, Mode mode, const SessionContextIntf* ctx)
  : SessionMember(ctx)
  , _txn(nullptr)
  , _mode(mode)
  , _done(false)
{
  unsigned flags = (mode == Mode::ReadOnly) ? MDB_RDONLY : 0u;
  int rc = mdb_txn_begin(env.raw(), nullptr, flags, &_txn);
  if (rc != MDB_SUCCESS) {
    _txn  = nullptr;
    _done = true;
    session()->error(fmt("mdb_txn_begin failed: {}", mdb_strerror(rc)));
  }
}

LmdbTxn::LmdbTxn(LmdbTxn&& other) noexcept
  : SessionMember(other.session())
  , _txn(other._txn)
  , _mode(other._mode)
  , _done(other._done)
{
  other._txn  = nullptr;
  other._done = true;
}

LmdbTxn&
LmdbTxn::operator=(LmdbTxn&& other) noexcept
{
  if (this == &other) {
    return *this;
  }
  if (_txn && !_done) {
    mdb_txn_abort(_txn);
  }
  _txn  = other._txn;
  _mode = other._mode;
  _done = other._done;
  other._txn  = nullptr;
  other._done = true;
  return *this;
}

LmdbTxn::~LmdbTxn()
{
  if (_txn && !_done) {
    mdb_txn_abort(_txn);
  }
}

void
LmdbTxn::commit()
{
  if (!_txn || _done) {
    session()->error(fmt("lmdb commit on inactive txn"));
    return;
  }
  MDB_txn* t = _txn;
  _txn  = nullptr;
  _done = true;
  int rc = mdb_txn_commit(t);
  if (rc != MDB_SUCCESS) {
    session()->error(fmt("mdb_txn_commit failed: {}", mdb_strerror(rc)));
  }
}

void
LmdbTxn::abort() noexcept
{
  if (_txn && !_done) {
    mdb_txn_abort(_txn);
  }
  _txn  = nullptr;
  _done = true;
}

bool
LmdbTxn::active() const noexcept
{
  return _txn != nullptr && !_done;
}

LmdbTxn::Mode
LmdbTxn::mode() const noexcept
{
  return _mode;
}

MDB_txn*
LmdbTxn::raw() const noexcept
{
  return _txn;
}

}
