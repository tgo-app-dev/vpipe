#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

using namespace std;

namespace vpipe {

namespace {

inline MDB_val
to_val(string_view s)
{
  return MDB_val{s.size(), const_cast<char*>(s.data())};
}

inline string_view
from_val(const MDB_val& v)
{
  return string_view(static_cast<const char*>(v.mv_data), v.mv_size);
}

}

LmdbDb::LmdbDb(LmdbEnv& env, string_view name, bool create_if_missing)
  : LmdbDb(env, env.session(), name, create_if_missing)
{
}

LmdbDb::LmdbDb(LmdbEnv&                  env,
               const SessionContextIntf* ctx,
               string_view               name,
               bool                      create_if_missing)
  : SessionMember(ctx)
  , _dbi(0)
  , _name(name)
{
  LmdbTxn boot(env, LmdbTxn::Mode::ReadWrite, ctx);
  unsigned flags = create_if_missing ? MDB_CREATE : 0u;
  int rc = mdb_dbi_open(boot.raw(), _name.c_str(), flags, &_dbi);
  if (rc != MDB_SUCCESS) {
    boot.abort();
    session()->error(
      fmt("mdb_dbi_open failed for '{}': {}", _name, mdb_strerror(rc)));
    return;
  }
  boot.commit();
}

LmdbDb::LmdbDb(LmdbDb&& other) noexcept
  : SessionMember(other.session())
  , _dbi(other._dbi)
  , _name(std::move(other._name))
{
  other._dbi = 0;
}

LmdbDb&
LmdbDb::operator=(LmdbDb&& other) noexcept
{
  if (this == &other) {
    return *this;
  }
  _dbi  = other._dbi;
  _name = std::move(other._name);
  other._dbi = 0;
  return *this;
}

optional<string_view>
LmdbDb::get(LmdbTxn& txn, string_view key) const
{
  MDB_val k = to_val(key);
  MDB_val v{};
  int rc = mdb_get(txn.raw(), _dbi, &k, &v);
  if (rc == MDB_NOTFOUND) {
    return nullopt;
  }
  if (rc != MDB_SUCCESS) {
    session()->error(
      fmt("mdb_get failed in '{}': {}", _name, mdb_strerror(rc)));
    return nullopt;
  }
  return from_val(v);
}

void
LmdbDb::put(LmdbTxn&    txn,
            string_view key,
            string_view value,
            unsigned    flags)
{
  MDB_val k = to_val(key);
  MDB_val v = to_val(value);
  int rc = mdb_put(txn.raw(), _dbi, &k, &v, flags);
  if (rc != MDB_SUCCESS) {
    session()->error(
      fmt("mdb_put failed in '{}': {}", _name, mdb_strerror(rc)));
  }
}

bool
LmdbDb::del(LmdbTxn& txn, string_view key)
{
  MDB_val k = to_val(key);
  int rc = mdb_del(txn.raw(), _dbi, &k, nullptr);
  if (rc == MDB_NOTFOUND) {
    return false;
  }
  if (rc != MDB_SUCCESS) {
    session()->error(
      fmt("mdb_del failed in '{}': {}", _name, mdb_strerror(rc)));
    return false;
  }
  return true;
}

MDB_dbi
LmdbDb::dbi() const noexcept
{
  return _dbi;
}

string_view
LmdbDb::name() const noexcept
{
  return _name;
}

}
