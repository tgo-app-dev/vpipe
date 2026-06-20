// vpipe-db-log-reader: prints records from a DbLogDelegate's LMDB
// "log" sub-database to stdout, optionally filtered by a [from, to]
// time range. With --cleanup, deletes records in the same range
// instead of printing -- by default the range is "everything older
// than 1 day", but --from / --to override either bound. The on-disk
// schema is owned by common/db-log-delegate.cc: 12-byte big-endian
// (ns_since_epoch, seq) keys; FlexData binary values with {level,
// ts_ns, msg}.

#include "common/flex-data.h"
#include "common/lmdb-cursor.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/vpipe-format.h"
#include "interfaces/log-delegate-intf.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

// Minimal SessionContextIntf for app-side use: error throws,
// warnings go to stderr, everything else is silenced. The pool /
// edge-capacity hooks aren't exercised on the read path.
class StderrSession : public vpipe::SessionContextIntf {
public:
  void error(const vpipe::VpipeFormat& f) const override {
    std::cerr << "[ERROR] " << f() << '\n';
    throw std::runtime_error(f());
  }
  void warn(const vpipe::VpipeFormat& f) const override {
    std::cerr << "[WARN] " << f() << '\n';
  }
  void info       (const vpipe::VpipeFormat&) const override {}
  void log_debug  (const vpipe::VpipeFormat&) const override {}
  void log_verbose(const vpipe::VpipeFormat&) const override {}
  void log_normal (const vpipe::VpipeFormat&) const override {}
  void log_always (const vpipe::VpipeFormat& f) const override {
    std::cerr << f() << '\n';
  }
  vpipe::ThreadPool* thread_pool() const noexcept override
  {
    return nullptr;
  }
  unsigned default_edge_capacity() const noexcept override
  {
    return 0;
  }
  const vpipe::FFmpegLibraries* ffmpeg_libraries() const override
  {
    // db-log-reader is a pure log dumper -- no pipeline ever runs in
    // this app. Return nullptr; nothing in this binary asks for it.
    return nullptr;
  }
  vpipe::LmdbEnv* lmdb_env() const override
  {
    // db-log-reader opens its own env directly via the LmdbEnv
    // wrapper rather than going through the session's accessor;
    // this hook isn't part of the program flow.
    return nullptr;
  }
  vpipe::CoreMLModelManager* coreml_model_manager() const override
  {
    return nullptr;
  }
  vpipe::genai::GenerativeModelManager*
  generative_model_manager() const override
  {
    return nullptr;
  }
};

bool
is_all_digits_(std::string_view s) noexcept
{
  if (s.empty()) {
    return false;
  }
  for (char c : s) {
    if (c < '0' || c > '9') {
      return false;
    }
  }
  return true;
}

// Parse a time string into nanoseconds since the Unix epoch.
// Accepted forms:
//   * RFC3339 / ISO 8601: YYYY-MM-DDTHH:MM:SS[.fraction][Z]   (UTC)
//   * Pure decimal integer: digit count selects the unit
//       <= 10 digits  -> seconds
//       11..13 digits -> milliseconds
//       14..16 digits -> microseconds
//       >= 17 digits  -> nanoseconds
bool
parse_time_ns_(std::string_view s, std::int64_t* out_ns) noexcept
{
  if (s.empty()) {
    return false;
  }
  if (is_all_digits_(s)) {
    std::int64_t v = 0;
    for (char c : s) {
      v = v * 10 + (c - '0');
    }
    if (s.size() <= 10) {
      v *= 1'000'000'000LL;
    } else if (s.size() <= 13) {
      v *= 1'000'000LL;
    } else if (s.size() <= 16) {
      v *= 1'000LL;
    }
    *out_ns = v;
    return true;
  }

  // ISO 8601 form. Use sscanf for the date-time prefix; parse the
  // optional fractional seconds manually.
  std::string buf(s);
  int Y = 0, M = 0, D = 0, h = 0, m = 0, sec = 0;
  int n = std::sscanf(buf.c_str(),
                      "%d-%d-%dT%d:%d:%d",
                      &Y, &M, &D, &h, &m, &sec);
  if (n != 6) {
    return false;
  }
  std::tm tm{};
  tm.tm_year = Y - 1900;
  tm.tm_mon  = M - 1;
  tm.tm_mday = D;
  tm.tm_hour = h;
  tm.tm_min  = m;
  tm.tm_sec  = sec;
  std::time_t t = ::timegm(&tm);
  if (t == static_cast<std::time_t>(-1)) {
    return false;
  }
  std::int64_t ns = static_cast<std::int64_t>(t) * 1'000'000'000LL;

  size_t dot = s.find('.');
  if (dot != std::string_view::npos) {
    std::string_view frac = s.substr(dot + 1);
    if (!frac.empty()
        && (frac.back() == 'Z' || frac.back() == 'z')) {
      frac.remove_suffix(1);
    }
    long long fv = 0;
    int consumed = 0;
    for (char c : frac) {
      if (consumed == 9) {
        break;
      }
      if (c < '0' || c > '9') {
        return false;
      }
      fv = fv * 10 + (c - '0');
      ++consumed;
    }
    while (consumed < 9) {
      fv *= 10;
      ++consumed;
    }
    ns += fv;
  }
  *out_ns = ns;
  return true;
}

// Format a nanosecond timestamp as RFC3339 UTC with 9-digit fraction.
std::string
format_ts_(std::int64_t ns)
{
  std::time_t sec = static_cast<std::time_t>(ns / 1'000'000'000LL);
  long long   frac = ns % 1'000'000'000LL;
  if (frac < 0) {
    frac += 1'000'000'000LL;
    sec  -= 1;
  }
  std::tm tm{};
  ::gmtime_r(&sec, &tm);
  char buf[40];
  std::snprintf(buf, sizeof(buf),
                "%04d-%02d-%02dT%02d:%02d:%02d.%09lldZ",
                tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                tm.tm_hour, tm.tm_min, tm.tm_sec, frac);
  return std::string(buf);
}

std::uint64_t
be64_decode_(const char* p) noexcept
{
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) {
    v = (v << 8) | static_cast<unsigned char>(p[i]);
  }
  return v;
}

// 12-byte seek key: be64(ns) || 4 zero bytes for the seq prefix.
std::string
make_start_key_(std::int64_t ns) noexcept
{
  std::uint64_t u = ns < 0 ? 0u : static_cast<std::uint64_t>(ns);
  std::string s(12, '\0');
  for (int i = 7; i >= 0; --i) {
    s[i] = static_cast<char>(u & 0xffu);
    u >>= 8;
  }
  return s;
}

void
usage_(const char* prog)
{
  std::fprintf(stderr,
    "Usage: %s <db-path> [--from TS] [--to TS] [--map-size-mb N]\n"
    "       %s <db-path> --cleanup [--from TS] [--to TS] [--dry-run]\n"
    "                    [--map-size-mb N]\n"
    "\n"
    "  TS is one of:\n"
    "    * RFC3339 / ISO 8601 in UTC, e.g. 2026-05-04T12:34:56Z\n"
    "      or 2026-05-04T12:34:56.123456789Z\n"
    "    * Pure integer count since the Unix epoch; the unit is\n"
    "      inferred from the digit count (s / ms / us / ns).\n"
    "\n"
    "  --map-size-mb N  override the LMDB map size (default 64).\n"
    "  --cleanup        delete records in [from, to] instead of\n"
    "                   printing. Default range is everything older\n"
    "                   than 1 day (i.e. --to defaults to now - 24h\n"
    "                   and --from defaults to 0). Prints a summary\n"
    "                   of records deleted.\n"
    "  --dry-run        with --cleanup, count what *would* be deleted\n"
    "                   without writing. No effect outside --cleanup.\n",
    prog, prog);
}

}

int
main(int argc, char** argv)
{
  std::string  db_path;
  std::int64_t from_ns      = 0;
  std::int64_t to_ns        = INT64_MAX;
  std::size_t  map_size     = std::size_t{1} << 26;   // 64 MiB
  bool         cleanup_mode = false;
  bool         dry_run      = false;
  bool         to_set       = false;

  for (int i = 1; i < argc; ++i) {
    std::string_view a = argv[i];
    if (a == "--from" && i + 1 < argc) {
      if (!parse_time_ns_(argv[++i], &from_ns)) {
        std::fprintf(stderr, "bad --from value: %s\n", argv[i]);
        return 2;
      }
    } else if (a == "--to" && i + 1 < argc) {
      if (!parse_time_ns_(argv[++i], &to_ns)) {
        std::fprintf(stderr, "bad --to value: %s\n", argv[i]);
        return 2;
      }
      to_set = true;
    } else if (a == "--map-size-mb" && i + 1 < argc) {
      char* end = nullptr;
      unsigned long long mb = std::strtoull(argv[++i], &end, 10);
      if (!end || *end != '\0' || mb == 0) {
        std::fprintf(stderr, "bad --map-size-mb value: %s\n",
                     argv[i]);
        return 2;
      }
      map_size = static_cast<std::size_t>(mb) << 20;
    } else if (a == "--cleanup") {
      cleanup_mode = true;
    } else if (a == "--dry-run") {
      dry_run = true;
    } else if (a == "--help" || a == "-h") {
      usage_(argv[0]);
      return 0;
    } else if (db_path.empty() && !a.empty() && a[0] != '-') {
      db_path = a;
    } else {
      std::fprintf(stderr, "unexpected argument: %s\n", argv[i]);
      usage_(argv[0]);
      return 2;
    }
  }
  if (db_path.empty()) {
    usage_(argv[0]);
    return 2;
  }
  if (dry_run && !cleanup_mode) {
    std::fprintf(stderr,
        "--dry-run only applies with --cleanup; ignoring.\n");
    dry_run = false;
  }
  if (cleanup_mode && !to_set) {
    // Default cleanup horizon: 1 day ago. Symmetric with the
    // VideosDbCleanupStage default so operators have one number to
    // remember for retention.
    const auto now_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
        .count();
    to_ns = static_cast<std::int64_t>(now_ns)
            - 86400LL * 1'000'000'000LL;
  }

  StderrSession sess;
  try {
    vpipe::LmdbEnv env(&sess, db_path, map_size);
    vpipe::LmdbDb  db(env, "log");

    if (!cleanup_mode) {
      // Print path: read-only walk of the [from, to] range.
      vpipe::LmdbTxn    txn(env, vpipe::LmdbTxn::Mode::ReadOnly);
      vpipe::LmdbCursor cur(txn, db);

      std::string start_key = make_start_key_(from_ns);
      std::string_view k, v;
      bool hit = cur.seek_at_or_after(start_key, k, v);

      while (hit) {
        if (k.size() < 12) {
          hit = cur.next(k, v);
          continue;
        }
        std::uint64_t ns_u = be64_decode_(k.data());
        std::int64_t  ns   = static_cast<std::int64_t>(ns_u);
        if (ns > to_ns) {
          break;
        }

        std::int64_t level = -1;
        std::int64_t ts    = ns;
        std::string  msg;
        try {
          vpipe::FlexData rec = vpipe::FlexData::from_binary(v);
          if (rec.is_object()) {
            auto obj = rec.as_object();
            if (obj.contains("level")) {
              level = obj.at("level").as_int(-1);
            }
            if (obj.contains("ts_ns")) {
              ts = static_cast<std::int64_t>(
                obj.at("ts_ns").as_uint(
                    static_cast<std::uint64_t>(ns)));
            }
            if (obj.contains("msg")) {
              msg = std::string(obj.at("msg").as_string(""));
            }
          }
        } catch (const std::exception& e) {
          // Malformed value: keep going, but make the line obvious.
          msg = std::string("<decode failed: ") + e.what() + '>';
        }

        const char* lvl = "?";
        if (level >= 0
            && level <= static_cast<std::int64_t>(vpipe::LogLevel::Always))
        {
          lvl = vpipe::to_cstr(static_cast<vpipe::LogLevel>(level));
        }

        std::cout << format_ts_(ts) << " [" << lvl << "] " << msg
                  << '\n';

        hit = cur.next(k, v);
      }
      txn.abort();
      return 0;
    }

    // Cleanup path. Two-pass: read pass collects keys to delete,
    // write pass removes them. Reading and deleting in one cursor
    // walk is fragile (LMDB invalidates cursor position on
    // mdb_del), and the read txn releases the writer mutex
    // immediately so any concurrent producer keeps appending.
    std::vector<std::string> to_delete;
    std::int64_t             oldest_ns = INT64_MAX;
    std::int64_t             newest_ns = INT64_MIN;
    {
      vpipe::LmdbTxn    txn(env, vpipe::LmdbTxn::Mode::ReadOnly);
      vpipe::LmdbCursor cur(txn, db);

      std::string start_key = make_start_key_(from_ns);
      std::string_view k, v;
      bool hit = cur.seek_at_or_after(start_key, k, v);
      while (hit) {
        if (k.size() < 12) {
          hit = cur.next(k, v);
          continue;
        }
        std::uint64_t ns_u = be64_decode_(k.data());
        std::int64_t  ns   = static_cast<std::int64_t>(ns_u);
        if (ns > to_ns) {
          break;
        }
        if (ns < oldest_ns) { oldest_ns = ns; }
        if (ns > newest_ns) { newest_ns = ns; }
        to_delete.emplace_back(k);  // copy bytes out of LMDB memory
        hit = cur.next(k, v);
      }
      txn.abort();
    }

    if (to_delete.empty()) {
      std::cout << "no records in [" << format_ts_(from_ns)
                << ", " << format_ts_(to_ns) << "] to delete\n";
      return 0;
    }

    if (dry_run) {
      std::cout << "[dry-run] would delete " << to_delete.size()
                << " record(s); range "
                << format_ts_(oldest_ns) << " .. "
                << format_ts_(newest_ns) << '\n';
      return 0;
    }

    std::size_t deleted = 0;
    {
      vpipe::LmdbTxn txn(env, vpipe::LmdbTxn::Mode::ReadWrite);
      for (const auto& k : to_delete) {
        if (db.del(txn, k)) {
          ++deleted;
        }
      }
      txn.commit();
    }
    std::cout << "deleted " << deleted << " record(s); range "
              << format_ts_(oldest_ns) << " .. "
              << format_ts_(newest_ns) << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
