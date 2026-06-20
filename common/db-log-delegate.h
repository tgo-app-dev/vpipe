#ifndef DB_LOG_DELEGATE_H
#define DB_LOG_DELEGATE_H

#include "common/lmdb-env.h"
#include "interfaces/log-delegate-intf.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace vpipe {

class LmdbDb;

// Persists log records into an LMDB-backed "log" sub-database. Each
// record is keyed by a 12-byte big-endian (ns_since_epoch, seq) tuple
// so a lexicographic scan walks records in chronological order, with
// the per-process counter breaking same-ns ties. Each value is the
// FlexData binary encoding of an object
//
//     { "level": <int>,  "ts_ns": <uint>,  "msg": <string> }
//
// The delegate does NOT own the LMDB env -- it borrows the
// session-shared LmdbEnv passed at construction (typically obtained
// via SessionContextIntf::lmdb_env()). The env must outlive the
// delegate; the Session enforces this by destroying the delegate
// before tearing down its own env.
//
// Recursion firewall: the delegate uses a private SessionContextIntf
// adapter for its own LmdbDb / per-call LmdbTxn so an LMDB error
// inside the delegate is reported to std::cerr and never re-enters
// the outer Session's delegate (which would otherwise be us,
// recursively).
//
// Thread safety: log() is safe to call concurrently. An internal
// mutex serialises the txn lifetime and the per-process sequence
// counter; LMDB itself serialises write txns globally per env so the
// mutex adds no real contention beyond what LMDB already imposes.
//
// TODO (Phase B): once the Session thread pool / job dispatcher
// lands, replace in-place writes with an enqueue onto an MPSC queue
// drained by a logging consumer that batches records into one
// MDB_APPEND-style txn per drain cycle. Producer side becomes
// non-blocking; the LogDelegateIntf surface does not change.
class DbLogDelegate final : public LogDelegateIntf {
public:
  // `env` is non-owning and must outlive this delegate. Construction
  // throws if the "log" sub-db cannot be opened in `env` (the env
  // itself is opened upstream by the Session).
  DbLogDelegate(LmdbEnv* env,
                LogLevel threshold = LogLevel::Normal);

  // Backwards-compatible alias for the env's default mmap size.
  // The session config's `db.map_size_mb` knob ultimately wins; this
  // is what gets used when no explicit size is configured.
  static constexpr std::size_t kDefaultMapSize = LmdbEnv::kDefaultMapSize;

  DbLogDelegate(const DbLogDelegate&)            = delete;
  DbLogDelegate& operator=(const DbLogDelegate&) = delete;

  ~DbLogDelegate() override;

  void log(LogLevel, const VpipeFormat&) override;

  // set_threshold() is only legal while no pipeline is running -- the
  // Session enforces this. So plain reads / writes here are sound.
  LogLevel threshold() const noexcept override { return _threshold; }
  void set_threshold(LogLevel level) override  { _threshold = level; }

private:
  class CerrSessionContext;

  std::unique_ptr<CerrSessionContext> _ctx;
  LmdbEnv*                            _env;  // non-owning
  std::unique_ptr<LmdbDb>             _db;
  std::mutex                          _mu;
  std::uint64_t                       _seq;
  LogLevel                            _threshold;
};

}

#endif
