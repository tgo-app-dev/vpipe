#include "common/lmdb-txn.h"
#include "common/lmdb-env.h"
#include "common/vpipe-format.h"
#include "interfaces/log-sink-intf.h"

namespace vpipe {

LmdbTxn::LmdbTxn(LmdbEnv& env, Mode mode)
  : LmdbTxn(env, mode, env.log())
{
}

LmdbTxn::LmdbTxn(LmdbEnv& env, Mode mode, const LogSinkIntf* ctx)
  : _log(ctx)
  , _txn(nullptr)
  , _mode(mode)
  , _done(false)
{
  unsigned flags = (mode == Mode::ReadOnly) ? MDB_RDONLY : 0u;
  int rc = mdb_txn_begin(env.raw(), nullptr, flags, &_txn);
  if (rc != MDB_SUCCESS) {
    _txn  = nullptr;
    _done = true;
    _log->error(fmt("mdb_txn_begin failed: {}", mdb_strerror(rc)));
  }
}

LmdbTxn::LmdbTxn(LmdbTxn&& other) noexcept
  : _log(other._log)
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
    _log->error(fmt("lmdb commit on inactive txn"));
    return;
  }
  MDB_txn* t = _txn;
  _txn  = nullptr;
  _done = true;
  int rc = mdb_txn_commit(t);
  if (rc != MDB_SUCCESS) {
    _log->error(fmt("mdb_txn_commit failed: {}", mdb_strerror(rc)));
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
