#include "common/lmdb-env.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include <filesystem>
#include <system_error>

using namespace std;

namespace vpipe {

void
lmdb_check(const SessionContextIntf* session,
           int                       rc,
           string_view               op)
{
  if (rc == MDB_SUCCESS) {
    return;
  }
  session->error(fmt("lmdb {} failed: {}", op, mdb_strerror(rc)));
}

LmdbEnv::LmdbEnv(const SessionContextIntf* session,
                 string_view               path,
                 size_t                    map_size,
                 unsigned                  max_dbs,
                 unsigned                  max_readers,
                 unsigned                  extra_flags)
  : SessionMember(session)
  , _env(nullptr)
  , _path(path)
{
  // On any failure: close env (if open) before routing through error().
  // The dtor doesn't run for a partially-constructed object after a throw.
  auto check = [&](int rc, const char* op) {
    if (rc == MDB_SUCCESS) {
      return;
    }
    if (_env) {
      mdb_env_close(_env);
      _env = nullptr;
    }
    this->session()->error(
      fmt("lmdb {} failed for {}: {}", op, _path, mdb_strerror(rc)));
  };

  if (!(extra_flags & MDB_NOSUBDIR)) {
    error_code ec;
    filesystem::create_directories(_path, ec);
    if (ec) {
      this->session()->error(
        fmt("create_directories failed for {}: {}", _path, ec.message()));
      return;
    }
  }

  int rc = mdb_env_create(&_env);
  if (rc != MDB_SUCCESS) {
    _env = nullptr;
  }
  check(rc, "mdb_env_create");

  check(mdb_env_set_mapsize(_env, map_size),    "mdb_env_set_mapsize");
  check(mdb_env_set_maxdbs(_env, max_dbs),      "mdb_env_set_maxdbs");
  check(mdb_env_set_maxreaders(_env, max_readers),
                                                "mdb_env_set_maxreaders");
  check(mdb_env_open(_env, _path.c_str(),
                     extra_flags | MDB_NOTLS, 0644),
                                                "mdb_env_open");
}

LmdbEnv::LmdbEnv(LmdbEnv&& other) noexcept
  : SessionMember(other.session())
  , _env(other._env)
  , _path(std::move(other._path))
{
  other._env = nullptr;
}

LmdbEnv&
LmdbEnv::operator=(LmdbEnv&& other) noexcept
{
  if (this == &other) {
    return *this;
  }
  if (_env) {
    mdb_env_close(_env);
  }
  _env  = other._env;
  _path = std::move(other._path);
  other._env = nullptr;
  return *this;
}

LmdbEnv::~LmdbEnv()
{
  if (_env) {
    mdb_env_close(_env);
  }
}

bool
LmdbEnv::valid() const noexcept
{
  return _env != nullptr;
}

string_view
LmdbEnv::path() const noexcept
{
  return _path;
}

MDB_env*
LmdbEnv::raw() const noexcept
{
  return _env;
}

}
