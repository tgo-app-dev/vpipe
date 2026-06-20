#ifndef LMDB_ENV_H
#define LMDB_ENV_H

#include "common/session-member.h"
#include <cstddef>
#include <lmdb.h>
#include <string>
#include <string_view>

namespace vpipe {

class SessionContextIntf;

// Routes an LMDB return code through the session's error path. On
// rc != MDB_SUCCESS, calls session->error(...) (which throws). Used by all
// lmdb-* wrappers.
void lmdb_check(const SessionContextIntf* session,
                int                       rc,
                std::string_view          op);

// RAII owner of an MDB_env*. One env per session is the intended pattern;
// multiple LmdbDb sub-databases live inside it.
//
// Thread-safety: after construction, an LmdbEnv reference may be shared
// freely across threads. The env is opened with MDB_NOTLS so read txns are
// not bound to the calling thread, which matches the upcoming Session
// thread-pool / job dispatcher. Destruction and move must be exclusive
// (no live txns or workers).
class LmdbEnv : public SessionMember {
public:
  // Default mmap size when the caller does not specify one. 1 GiB --
  // generous for a session-shared env, keeps room for the log
  // sub-db plus any user sub-dbs without resizing.
  static constexpr std::size_t kDefaultMapSize = std::size_t{1} << 30;

  // Creates the directory if it doesn't exist (unless MDB_NOSUBDIR is in
  // extra_flags) and opens / creates the env. Errors via session->error()
  // (throws). MDB_NOTLS is always OR'd into the open flags; extra_flags is
  // for additional options (e.g. MDB_NOSUBDIR, MDB_NOSYNC). max_readers
  // sizes the reader-slot table; with NOTLS each *active* read txn consumes
  // one slot, so size it >= the worker pool's expected concurrent readers.
  LmdbEnv(const SessionContextIntf* session,
          std::string_view          path,
          std::size_t               map_size    = kDefaultMapSize,
          unsigned                  max_dbs     = 16,
          unsigned                  max_readers = 256,
          unsigned                  extra_flags = 0);

  LmdbEnv(const LmdbEnv&)            = delete;
  LmdbEnv& operator=(const LmdbEnv&) = delete;
  LmdbEnv(LmdbEnv&&) noexcept;
  LmdbEnv& operator=(LmdbEnv&&) noexcept;
  ~LmdbEnv() override;

  bool             valid() const noexcept;
  std::string_view path()  const noexcept;
  MDB_env*         raw()   const noexcept;
  explicit operator bool() const noexcept { return valid(); }

private:
  MDB_env*    _env;
  std::string _path;
};

}

#endif
