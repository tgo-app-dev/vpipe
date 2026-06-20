#ifndef LMDB_CURSOR_H
#define LMDB_CURSOR_H

#include "common/session-member.h"
#include <lmdb.h>
#include <string_view>

namespace vpipe {

class SessionContextIntf;
class LmdbTxn;
class LmdbDb;

// RAII over MDB_cursor*. Iteration / range scans for a given dbi within a
// txn. Bound to its parent txn for the duration of any cursor op.
//
// All step / seek methods return true on hit; the out views point into
// LMDB-managed memory valid until the next cursor op or txn end. End-of-
// range returns false. Real LMDB errors throw via session()->error().
class LmdbCursor : public SessionMember {
public:
  LmdbCursor(LmdbTxn& txn, const LmdbDb& db);

  LmdbCursor(const LmdbCursor&)            = delete;
  LmdbCursor& operator=(const LmdbCursor&) = delete;
  LmdbCursor(LmdbCursor&&) noexcept;
  LmdbCursor& operator=(LmdbCursor&&) noexcept;
  ~LmdbCursor() override;

  bool first(std::string_view& k, std::string_view& v);
  bool last (std::string_view& k, std::string_view& v);
  bool next (std::string_view& k, std::string_view& v);
  bool prev (std::string_view& k, std::string_view& v);

  // MDB_SET_RANGE: position at the first key >= key_in.
  bool seek_at_or_after(std::string_view  key_in,
                        std::string_view& k_out,
                        std::string_view& v_out);

  MDB_cursor* raw() const noexcept;

private:
  MDB_cursor* _cur;
};

}

#endif
