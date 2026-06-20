#include "common/db-log-delegate.h"
#include "common/flex-data.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace vpipe {

// Private SessionContextIntf adapter used only inside the delegate.
// Routes LMDB errors / warnings to std::cerr (and throws on error,
// which is what the LMDB wrappers expect). Critically it does NOT
// route through the outer Session's delegate, breaking the recursion
// path session.error -> delegate -> LMDB write fails -> session.error
// -> ... structurally.
class DbLogDelegate::CerrSessionContext final
  : public SessionContextIntf
{
public:
  void error(const VpipeFormat& f) const override {
    std::cerr << "[ERROR] [vpipe-log] " << f() << '\n';
    std::cerr.flush();
    throw std::runtime_error("[vpipe-log] " + f());
  }
  void warn(const VpipeFormat& f) const override {
    std::cerr << "[WARN] [vpipe-log] " << f() << '\n';
    std::cerr.flush();
  }
  void info(const VpipeFormat& f) const override {
    std::cerr << "[INFO] [vpipe-log] " << f() << '\n';
  }
  void log_debug(const VpipeFormat& f) const override {
    std::cerr << "[DEBUG] [vpipe-log] " << f() << '\n';
  }
  void log_verbose(const VpipeFormat& f) const override {
    std::cerr << "[VERBOSE] [vpipe-log] " << f() << '\n';
  }
  void log_normal(const VpipeFormat& f) const override {
    std::cerr << "[NORMAL] [vpipe-log] " << f() << '\n';
  }
  void log_always(const VpipeFormat& f) const override {
    std::cerr << "[ALWAYS] [vpipe-log] " << f() << '\n';
  }
  // Bootstrap context only services logging during DbLogDelegate's
  // own LMDB operations; pipelines never see this context, so the
  // pool / capacity / ffmpeg / env hooks are not exercised from
  // here.
  ThreadPool* thread_pool() const noexcept override {
    return nullptr;
  }
  unsigned default_edge_capacity() const noexcept override {
    return 0;
  }
  const FFmpegLibraries* ffmpeg_libraries() const override {
    // No pipeline ever runs through this context, so no caller
    // should ask for FFmpeg. Surface nullptr rather than throwing
    // -- pointer return makes that the natural way to express
    // "not available here".
    return nullptr;
  }
  LmdbEnv* lmdb_env() const override {
    // The delegate already holds the env it needs; this hook only
    // exists for downstream SessionMembers asking the *session*
    // for its env. Surface a clear error if exercised.
    throw std::logic_error(
      "DbLogDelegate bootstrap context: lmdb_env() not "
      "available");
  }
  CoreMLModelManager* coreml_model_manager() const override {
    // No pipeline ever runs through this context.
    return nullptr;
  }
  genai::GenerativeModelManager* generative_model_manager() const override {
    // No pipeline ever runs through this context.
    return nullptr;
  }
};

namespace {

// Big-endian encode so lexicographic byte order matches numeric
// order. The 12-byte key is: be64(ns) || be32(seq).
void
be64_into(char* dst, std::uint64_t v)
{
  for (int i = 7; i >= 0; --i) {
    dst[i] = static_cast<char>(v & 0xffu);
    v >>= 8;
  }
}

void
be32_into(char* dst, std::uint32_t v)
{
  for (int i = 3; i >= 0; --i) {
    dst[i] = static_cast<char>(v & 0xffu);
    v >>= 8;
  }
}

std::string
make_key(std::uint64_t ns, std::uint64_t seq)
{
  std::string k(12, '\0');
  be64_into(k.data(), ns);
  be32_into(k.data() + 8, static_cast<std::uint32_t>(seq & 0xffffffffu));
  return k;
}

std::string
encode_record(LogLevel level, std::uint64_t ns, const std::string& msg)
{
  auto rec = FlexData::make_object();
  auto obj = rec.as_object();
  obj.insert("level", FlexData::make_int(static_cast<int64_t>(level)));
  obj.insert("ts_ns", FlexData::make_uint(ns));
  obj.insert("msg",   FlexData::make_string(msg));
  return rec.to_binary();
}

}

DbLogDelegate::DbLogDelegate(LmdbEnv* env, LogLevel threshold)
  : _ctx(std::make_unique<CerrSessionContext>())
  , _env(env)
  , _seq(0)
  , _threshold(threshold)
{
  if (!_env) {
    throw std::runtime_error(
        "DbLogDelegate: env is null (caller must supply an open env)");
  }
  // Construction errors propagate (caught by Session bootstrap).
  // The "log" sub-db is opened against the borrowed env using our
  // own context for the recursion firewall; runtime LMDB errors
  // surface on cerr instead of re-entering the active session
  // delegate.
  _db = std::make_unique<LmdbDb>(*_env, _ctx.get(), "log");
}

DbLogDelegate::~DbLogDelegate() = default;

void
DbLogDelegate::log(LogLevel level, const VpipeFormat& f)
{
  if (level != LogLevel::Always &&
      static_cast<int>(level) > static_cast<int>(_threshold)) {
    return;
  }

  // Format outside the mutex; LMDB write happens under it.
  std::string msg;
  try {
    msg = f();
  } catch (...) {
    msg = "<formatter threw>";
  }

  const auto ts_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());

  std::string key;
  std::string val;
  try {
    val = encode_record(level, ts_ns, msg);
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] [vpipe-log] flex encode failed: "
              << e.what() << '\n';
    std::cerr.flush();
    return;
  }

  try {
    std::lock_guard<std::mutex> lk(_mu);
    key = make_key(ts_ns, _seq++);
    LmdbTxn txn(*_env, LmdbTxn::Mode::ReadWrite, _ctx.get());
    _db->put(txn, key, val);
    txn.commit();
  } catch (const std::exception& e) {
    // LMDB write failed; report on cerr and swallow so the producer
    // doesn't see a logging exception.
    std::cerr << "[ERROR] [vpipe-log] write failed (level="
              << to_cstr(level) << "): " << e.what() << '\n';
    std::cerr << "[ERROR] [vpipe-log] dropped message: " << msg
              << '\n';
    std::cerr.flush();
  } catch (...) {
    std::cerr << "[ERROR] [vpipe-log] write failed (unknown)\n";
    std::cerr.flush();
  }
}

}
