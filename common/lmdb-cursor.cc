#include "common/lmdb-cursor.h"
#include "common/lmdb-db.h"
#include "common/lmdb-txn.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

using namespace std;

namespace vpipe {

namespace {

inline string_view
from_val(const MDB_val& v)
{
  return string_view(static_cast<const char*>(v.mv_data), v.mv_size);
}

}

LmdbCursor::LmdbCursor(LmdbTxn& txn, const LmdbDb& db)
  : SessionMember(txn.session())
  , _cur(nullptr)
{
  int rc = mdb_cursor_open(txn.raw(), db.dbi(), &_cur);
  if (rc != MDB_SUCCESS) {
    _cur = nullptr;
    session()->error(fmt("mdb_cursor_open failed: {}", mdb_strerror(rc)));
  }
}

LmdbCursor::LmdbCursor(LmdbCursor&& other) noexcept
  : SessionMember(other.session())
  , _cur(other._cur)
{
  other._cur = nullptr;
}

LmdbCursor&
LmdbCursor::operator=(LmdbCursor&& other) noexcept
{
  if (this == &other) {
    return *this;
  }
  if (_cur) {
    mdb_cursor_close(_cur);
  }
  _cur = other._cur;
  other._cur = nullptr;
  return *this;
}

LmdbCursor::~LmdbCursor()
{
  if (_cur) {
    mdb_cursor_close(_cur);
  }
}

namespace {

bool
step(const SessionContextIntf* s,
     MDB_cursor*               cur,
     MDB_cursor_op             op,
     MDB_val&                  k,
     MDB_val&                  v,
     string_view&              k_out,
     string_view&              v_out)
{
  int rc = mdb_cursor_get(cur, &k, &v, op);
  if (rc == MDB_NOTFOUND) {
    return false;
  }
  if (rc != MDB_SUCCESS) {
    s->error(fmt("mdb_cursor_get failed: {}", mdb_strerror(rc)));
    return false;
  }
  k_out = from_val(k);
  v_out = from_val(v);
  return true;
}

}

bool
LmdbCursor::first(string_view& k, string_view& v)
{
  MDB_val ki{}, vi{};
  return step(session(), _cur, MDB_FIRST, ki, vi, k, v);
}

bool
LmdbCursor::last(string_view& k, string_view& v)
{
  MDB_val ki{}, vi{};
  return step(session(), _cur, MDB_LAST, ki, vi, k, v);
}

bool
LmdbCursor::next(string_view& k, string_view& v)
{
  MDB_val ki{}, vi{};
  return step(session(), _cur, MDB_NEXT, ki, vi, k, v);
}

bool
LmdbCursor::prev(string_view& k, string_view& v)
{
  MDB_val ki{}, vi{};
  return step(session(), _cur, MDB_PREV, ki, vi, k, v);
}

bool
LmdbCursor::seek_at_or_after(string_view  key_in,
                             string_view& k_out,
                             string_view& v_out)
{
  MDB_val ki{key_in.size(), const_cast<char*>(key_in.data())};
  MDB_val vi{};
  return step(session(), _cur, MDB_SET_RANGE, ki, vi, k_out, v_out);
}

MDB_cursor*
LmdbCursor::raw() const noexcept
{
  return _cur;
}

}
