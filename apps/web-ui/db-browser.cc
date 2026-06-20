#include "apps/web-ui/db-browser.h"

#include "common/lmdb-env.h"
#include "interfaces/session-context-intf.h"

#include <lmdb.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <optional>
#include <string>
#include <string_view>

using namespace std;

namespace vpipe::webui {

namespace {

constexpr size_t kScanCap = 500000;   // bound work per keys query

// ---- base64 (standard alphabet) ----------------------------------
const char kB64[] =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

string
b64_encode(string_view in)
{
  string out;
  out.reserve(((in.size() + 2) / 3) * 4);
  size_t i = 0;
  auto uc = [](char c) { return static_cast<unsigned>((unsigned char)c); };
  for (; i + 3 <= in.size(); i += 3) {
    unsigned n = (uc(in[i]) << 16) | (uc(in[i + 1]) << 8) | uc(in[i + 2]);
    out += kB64[(n >> 18) & 63]; out += kB64[(n >> 12) & 63];
    out += kB64[(n >> 6) & 63];  out += kB64[n & 63];
  }
  if (in.size() - i == 1) {
    unsigned n = uc(in[i]) << 16;
    out += kB64[(n >> 18) & 63]; out += kB64[(n >> 12) & 63]; out += "==";
  } else if (in.size() - i == 2) {
    unsigned n = (uc(in[i]) << 16) | (uc(in[i + 1]) << 8);
    out += kB64[(n >> 18) & 63]; out += kB64[(n >> 12) & 63];
    out += kB64[(n >> 6) & 63];  out += "=";
  }
  return out;
}

optional<string>
b64_decode(string_view in)
{
  auto val = [](char c) -> int {
    if (c >= 'A' && c <= 'Z') { return c - 'A'; }
    if (c >= 'a' && c <= 'z') { return c - 'a' + 26; }
    if (c >= '0' && c <= '9') { return c - '0' + 52; }
    if (c == '+') { return 62; }
    if (c == '/') { return 63; }
    return -1;
  };
  string out;
  int buf = 0, bits = 0;
  for (char c : in) {
    if (c == '=' || c == '\n' || c == '\r' || c == ' ') { continue; }
    int d = val(c);
    if (d < 0) { return nullopt; }
    buf = (buf << 6) | d;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<char>((buf >> bits) & 0xFF));
    }
  }
  return out;
}

// Printable-ASCII display of arbitrary key bytes (so the JSON stays
// valid UTF-8); non-printable / high bytes shown as \xNN, capped.
string
escape_text(string_view s)
{
  string out;
  const size_t lim = std::min(s.size(), size_t{512});
  for (size_t i = 0; i < lim; ++i) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    if (c == '\t') { out += "\\t"; }
    else if (c == '\n') { out += "\\n"; }
    else if (c == '\r') { out += "\\r"; }
    else if (c >= 0x20 && c < 0x7f) { out += static_cast<char>(c); }
    else { char b[5]; snprintf(b, sizeof b, "\\x%02x", c); out += b; }
  }
  if (s.size() > lim) { out += "…"; }
  return out;
}

// Parse a key as a number: ASCII decimal/float consuming the whole
// string, else a big-endian unsigned integer for 1/2/4/8-byte keys.
optional<double>
key_as_number(string_view s)
{
  if (s.empty()) { return nullopt; }
  bool ascii_num = true, has_digit = false;
  for (char c : s) {
    unsigned char u = static_cast<unsigned char>(c);
    if (u >= '0' && u <= '9') { has_digit = true; }
    else if (!(c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E'
               || c == ' ')) { ascii_num = false; break; }
  }
  if (ascii_num && has_digit) {
    string tmp(s);
    char* end = nullptr;
    double d = strtod(tmp.c_str(), &end);
    while (end && *end == ' ') { ++end; }
    if (end && *end == '\0') { return d; }
  }
  if (s.size() == 1 || s.size() == 2 || s.size() == 4 || s.size() == 8) {
    unsigned long long v = 0;
    for (char c : s) { v = (v << 8) | static_cast<unsigned char>(c); }
    return static_cast<double>(v);
  }
  return nullopt;
}

// Scale a bare epoch number to seconds by magnitude (s/ms/us/ns).
double
epoch_scale(double n)
{
  double a = fabs(n);
  if (a >= 1e18) { return n / 1e9; }
  if (a >= 1e15) { return n / 1e6; }
  if (a >= 1e12) { return n / 1e3; }
  return n;
}

// Parse a user-entered date/time (UTC) to epoch seconds. Accepts a
// bare epoch number (auto-scaled) and a range of ISO-ish formats.
optional<double>
parse_datetime(string_view in)
{
  size_t b = 0, e = in.size();
  while (b < e && isspace((unsigned char)in[b])) { ++b; }
  while (e > b && isspace((unsigned char)in[e - 1])) { --e; }
  string s(in.substr(b, e - b));
  if (s.empty()) { return nullopt; }

  {
    bool num = true, dig = false;
    for (char c : s) {
      unsigned char u = static_cast<unsigned char>(c);
      if (u >= '0' && u <= '9') { dig = true; }
      else if (!(c == '-' || c == '+' || c == '.')) { num = false; break; }
    }
    if (num && dig) {
      char* end = nullptr;
      double d = strtod(s.c_str(), &end);
      if (end && *end == '\0') { return epoch_scale(d); }
    }
  }
  if (!s.empty() && (s.back() == 'Z' || s.back() == 'z')) { s.pop_back(); }

  static const char* fmts[] = {
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d",          "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",    "%Y/%m/%d",
  };
  for (const char* f : fmts) {
    struct tm tm{};
    char* r = strptime(s.c_str(), f, &tm);
    if (r && *r == '\0') { return static_cast<double>(timegm(&tm)); }
  }
  return nullopt;
}

optional<double>
key_as_time(string_view s)
{
  if (auto n = key_as_number(s)) { return epoch_scale(*n); }
  return parse_datetime(s);
}

string
format_time(double sec)
{
  time_t t = static_cast<time_t>(floor(sec));
  struct tm tm{};
  gmtime_r(&t, &tm);
  char buf[32];
  strftime(buf, sizeof buf, "%Y-%m-%d %H:%M:%S", &tm);
  string out = buf;
  double frac = sec - floor(sec);
  if (frac > 1e-6) {
    char fb[8];
    snprintf(fb, sizeof fb, ".%03d", static_cast<int>(frac * 1000 + 0.5));
    out += fb;
  }
  out += "Z";
  return out;
}

string
key_display(string_view k, const string& mode)
{
  if (mode == "number") {
    if (auto n = key_as_number(k)) {
      if (*n == floor(*n) && fabs(*n) < 9e15) {
        char b[32]; snprintf(b, sizeof b, "%lld", (long long)*n); return b;
      }
      char b[32]; snprintf(b, sizeof b, "%g", *n); return b;
    }
    return escape_text(k);
  }
  if (mode == "time") {
    if (auto t = key_as_time(k)) { return format_time(*t); }
    return escape_text(k);
  }
  return escape_text(k);
}

// ~1990 .. 2100 in epoch seconds.
bool
looks_like_epoch(double n)
{
  double s = epoch_scale(n);
  return s >= 6.3e8 && s <= 4.1e9;
}

string
detect_mode(string_view k)
{
  if (k.empty()) { return "text"; }
  bool ascii = true, has_digit = false;
  for (unsigned char c : k) {
    if (c < 0x20 || c >= 0x7f) { ascii = false; }
    if (c >= '0' && c <= '9') { has_digit = true; }
  }
  if (ascii) {
    // YYYY-MM-DD... / YYYY/MM/DD... -> time.
    if (k.size() >= 10 && isdigit((unsigned char)k[0])
        && isdigit((unsigned char)k[3])
        && (k[4] == '-' || k[4] == '/')) {
      return "time";
    }
    // ASCII numeric: epoch-magnitude -> time; small counters -> number.
    if (has_digit) {
      string s(k);
      char* end = nullptr;
      double d = strtod(s.c_str(), &end);
      if (end && *end == '\0') {
        return looks_like_epoch(d) ? "time" : "number";
      }
    }
    return "text";
  }
  // Binary key (e.g. the 8-byte big-endian us-since-epoch key the
  // video-segment writer uses). If the BE-integer reading lands in a
  // plausible real-world epoch range, treat as time; otherwise text.
  if (auto n = key_as_number(k); n && looks_like_epoch(*n)) {
    return "time";
  }
  return "text";
}

// od -t x1 -v -w16 -Ax of the first `maxlen` bytes.
string
hexdump(string_view data, size_t maxlen)
{
  const size_t n = std::min(data.size(), maxlen);
  string out;
  char buf[16];
  size_t off = 0;
  for (; off < n; off += 16) {
    snprintf(buf, sizeof buf, "%06zx", off);
    out += buf;
    size_t end = std::min(off + 16, n);
    for (size_t i = off; i < end; ++i) {
      snprintf(buf, sizeof buf, " %02x", (unsigned char)data[i]);
      out += buf;
    }
    out += "\n";
  }
  snprintf(buf, sizeof buf, "%06zx", n);
  out += buf;
  out += "\n";
  return out;
}

string
fstr_get(const FlexData::ConstObjectView& o, const char* key,
         const char* def = "")
{
  return o.contains(key) ? string(o.at(key).as_string(def)) : string(def);
}

}  // namespace

FlexData
DbBrowser::list_databases(string& err)
{
  LmdbEnv* env = _sctx ? _sctx->lmdb_env() : nullptr;
  if (!env || !env->valid()) {
    err = "no database available (lmdb_env)";
    return FlexData::make_object();
  }
  MDB_env* e = env->raw();
  MDB_txn* txn = nullptr;
  if (mdb_txn_begin(e, nullptr, MDB_RDONLY, &txn) != 0) {
    err = "could not open a read transaction";
    return FlexData::make_object();
  }
  MDB_dbi main = 0;
  if (mdb_dbi_open(txn, nullptr, 0, &main) != 0) {
    mdb_txn_abort(txn);
    err = "could not open the main database";
    return FlexData::make_object();
  }
  MDB_cursor* cur = nullptr;
  mdb_cursor_open(txn, main, &cur);

  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();
  MDB_val k, v;
  int rc = mdb_cursor_get(cur, &k, &v, MDB_FIRST);
  while (rc == 0) {
    string name(static_cast<const char*>(k.mv_data), k.mv_size);
    MDB_dbi sub = 0;
    if (!name.empty() && mdb_dbi_open(txn, name.c_str(), 0, &sub) == 0) {
      MDB_stat st;
      mdb_stat(txn, sub, &st);
      FlexData o = FlexData::make_object();
      auto oo = o.as_object();
      oo.insert("name", FlexData::make_string(name));
      oo.insert("count", FlexData::make_uint(st.ms_entries));
      a.push_back(std::move(o));
    }
    rc = mdb_cursor_get(cur, &k, &v, MDB_NEXT);
  }
  mdb_cursor_close(cur);
  mdb_txn_abort(txn);

  FlexData doc = FlexData::make_object();
  doc.as_object().insert("databases", std::move(arr));
  return doc;
}

FlexData
DbBrowser::query_keys(const FlexData& req, string& err)
{
  if (!req.is_object()) { err = "expected a JSON object"; return FlexData::make_object(); }
  auto root = req.as_object();
  const string dbname = fstr_get(root, "db");
  string mode  = fstr_get(root, "mode", "auto");
  const string match = fstr_get(root, "match", "exact");
  const string q  = fstr_get(root, "q");
  const string lo = fstr_get(root, "lo");
  const string hi = fstr_get(root, "hi");
  // Optional value-field filter: keep only keys whose value decodes to a
  // FlexData object with `value_field` == `value_equals`. Used by typed
  // model-registry suggestions (value_field="model_type"). Off when
  // value_field is empty.
  const string value_field  = fstr_get(root, "value_field");
  const string value_equals = fstr_get(root, "value_equals");
  const bool   value_filter = !value_field.empty();
  long page = root.contains("page") ? root.at("page").as_int(0) : 0;
  if (page < 0) { page = 0; }
  if (dbname.empty()) { err = "no database selected"; return FlexData::make_object(); }

  LmdbEnv* env = _sctx ? _sctx->lmdb_env() : nullptr;
  if (!env || !env->valid()) { err = "no database available"; return FlexData::make_object(); }
  MDB_env* e = env->raw();
  MDB_txn* txn = nullptr;
  if (mdb_txn_begin(e, nullptr, MDB_RDONLY, &txn) != 0) {
    err = "could not open a read transaction"; return FlexData::make_object();
  }
  MDB_dbi dbi = 0;
  if (mdb_dbi_open(txn, dbname.c_str(), 0, &dbi) != 0) {
    mdb_txn_abort(txn);
    err = "no such database: " + dbname; return FlexData::make_object();
  }
  MDB_cursor* cur = nullptr;
  mdb_cursor_open(txn, dbi, &cur);

  // Resolve auto mode from the first key.
  if (mode == "auto") {
    MDB_val k0, v0;
    if (mdb_cursor_get(cur, &k0, &v0, MDB_FIRST) == 0) {
      mode = detect_mode(string_view(static_cast<const char*>(k0.mv_data),
                                     k0.mv_size));
    } else {
      mode = "text";
    }
  }
  if (mode != "text" && mode != "number" && mode != "time") { mode = "text"; }

  // Pre-parse the query bounds for the resolved mode.
  const bool range = (match == "range");
  const bool match_all = range ? (lo.empty() && hi.empty()) : q.empty();
  optional<double> qn, lon, hin, qt, lot, hit;
  if (mode == "number") {
    if (!q.empty())  { qn  = key_as_number(q); }
    if (!lo.empty()) { lon = key_as_number(lo); }
    if (!hi.empty()) { hin = key_as_number(hi); }
  } else if (mode == "time") {
    if (!q.empty())  { qt  = parse_datetime(q); }
    if (!lo.empty()) { lot = parse_datetime(lo); }
    if (!hi.empty()) { hit = parse_datetime(hi); }
  }

  auto matches = [&](string_view ks) -> bool {
    if (match_all) { return true; }
    if (mode == "number") {
      auto kn = key_as_number(ks);
      if (!kn) { return false; }
      if (!range) { return qn && *kn == *qn; }
      return (!lon || *kn >= *lon) && (!hin || *kn <= *hin);
    }
    if (mode == "time") {
      auto kt = key_as_time(ks);
      if (!kt) { return false; }
      if (!range) { return qt && fabs(*kt - *qt) < 0.5; }
      return (!lot || *kt >= *lot) && (!hit || *kt <= *hit);
    }
    // text
    if (!range) { return ks == q; }
    return (lo.empty() || ks >= lo) && (hi.empty() || ks <= hi);
  };

  // Decode the value and test the optional value-field filter. Cheap when
  // off; only the matched-key set pays the decode. Non-FlexData / missing-
  // field values are dropped.
  auto value_ok = [&](const MDB_val& vv) -> bool {
    if (!value_filter) { return true; }
    try {
      string bytes(static_cast<const char*>(vv.mv_data), vv.mv_size);
      FlexData rec = FlexData::from_binary(bytes);
      if (!rec.is_object()) { return false; }
      auto obj = rec.as_object();
      if (!obj.contains(value_field)) { return false; }
      return string(obj.at(value_field).as_string("")) == value_equals;
    } catch (...) {
      return false;
    }
  };

  const long skip = page * DbBrowser::kPageSize;
  long matched = 0;
  bool truncated = false;
  size_t scanned = 0;
  FlexData keysArr = FlexData::make_array();
  auto ka = keysArr.as_array();

  MDB_val k, v;
  int rc = mdb_cursor_get(cur, &k, &v, MDB_FIRST);
  // Scan past the requested page so we know the total match count
  // (and thus the last page, for the First/Last buttons). Bounded by
  // kScanCap; for unfiltered browse on small/medium DBs the scan is
  // fast (memcmp), and the cap protects against runaway.
  while (rc == 0) {
    if (++scanned > kScanCap) { truncated = true; break; }
    string_view ks(static_cast<const char*>(k.mv_data), k.mv_size);
    if (matches(ks) && value_ok(v)) {
      long idx = matched++;
      if (idx >= skip
          && static_cast<long>(ka.size()) < DbBrowser::kPageSize) {
        FlexData o = FlexData::make_object();
        auto oo = o.as_object();
        oo.insert("key", FlexData::make_string(b64_encode(ks)));
        oo.insert("display", FlexData::make_string(key_display(ks, mode)));
        if (mode == "time") {
          if (auto t = key_as_time(ks)) {
            oo.insert("epoch", FlexData::make_real(*t));
          }
        }
        ka.push_back(std::move(o));
      }
    }
    rc = mdb_cursor_get(cur, &k, &v, MDB_NEXT);
  }
  mdb_cursor_close(cur);
  mdb_txn_abort(txn);

  const long total = matched;
  const long last_page = total > 0 ? (total - 1) / DbBrowser::kPageSize : 0;

  FlexData doc = FlexData::make_object();
  auto d = doc.as_object();
  d.insert("db", FlexData::make_string(dbname));
  d.insert("mode", FlexData::make_string(mode));
  d.insert("match", FlexData::make_string(range ? "range" : "exact"));
  d.insert("page", FlexData::make_uint(static_cast<uint64_t>(page)));
  d.insert("page_size", FlexData::make_uint(DbBrowser::kPageSize));
  d.insert("total", FlexData::make_uint(static_cast<uint64_t>(total)));
  d.insert("last_page",
           FlexData::make_uint(static_cast<uint64_t>(last_page)));
  d.insert("has_prev", FlexData::make_bool(page > 0));
  d.insert("has_next", FlexData::make_bool(page < last_page));
  d.insert("truncated", FlexData::make_bool(truncated));
  d.insert("keys", std::move(keysArr));
  return doc;
}

FlexData
DbBrowser::read_value(const FlexData& req, string& err)
{
  if (!req.is_object()) { err = "expected a JSON object"; return FlexData::make_object(); }
  auto root = req.as_object();
  const string dbname = fstr_get(root, "db");
  const string keyb64 = fstr_get(root, "key");
  if (dbname.empty()) { err = "no database selected"; return FlexData::make_object(); }
  auto keyOpt = b64_decode(keyb64);
  if (!keyOpt) { err = "bad key encoding"; return FlexData::make_object(); }
  const string key = *keyOpt;

  LmdbEnv* env = _sctx ? _sctx->lmdb_env() : nullptr;
  if (!env || !env->valid()) { err = "no database available"; return FlexData::make_object(); }
  MDB_env* e = env->raw();
  MDB_txn* txn = nullptr;
  if (mdb_txn_begin(e, nullptr, MDB_RDONLY, &txn) != 0) {
    err = "could not open a read transaction"; return FlexData::make_object();
  }
  MDB_dbi dbi = 0;
  if (mdb_dbi_open(txn, dbname.c_str(), 0, &dbi) != 0) {
    mdb_txn_abort(txn);
    err = "no such database: " + dbname; return FlexData::make_object();
  }
  MDB_val mk, mv;
  mk.mv_size = key.size();
  mk.mv_data = const_cast<char*>(key.data());
  int rc = mdb_get(txn, dbi, &mk, &mv);

  FlexData doc = FlexData::make_object();
  auto d = doc.as_object();
  if (rc == MDB_NOTFOUND) {
    d.insert("found", FlexData::make_bool(false));
    mdb_txn_abort(txn);
    return doc;
  }
  if (rc != 0) {
    mdb_txn_abort(txn);
    err = "read failed"; return FlexData::make_object();
  }
  string_view val(static_cast<const char*>(mv.mv_data), mv.mv_size);

  string encoding = "hex";
  string text;
  bool looks_flex = val.size() >= 8 && val[0] == 'V' && val[1] == 'P'
                    && val[2] == 'F' && val[3] == 'D';
  if (looks_flex) {
    try {
      FlexData fd = FlexData::from_binary(val);
      text = fd.to_json(/*pretty=*/true);
      encoding = "json";
    } catch (...) {
      looks_flex = false;
    }
  }
  if (!looks_flex) {
    text = hexdump(val, 128);
    encoding = "hex";
  }

  d.insert("found", FlexData::make_bool(true));
  d.insert("size", FlexData::make_uint(val.size()));
  d.insert("encoding", FlexData::make_string(encoding));
  d.insert("text", FlexData::make_string(text));
  d.insert("display", FlexData::make_string(escape_text(key)));
  mdb_txn_abort(txn);
  return doc;
}

FlexData
DbBrowser::delete_key(const FlexData& req, string& err)
{
  if (!req.is_object()) { err = "expected a JSON object"; return FlexData::make_object(); }
  auto root = req.as_object();
  const string dbname = fstr_get(root, "db");
  const string keyb64 = fstr_get(root, "key");
  if (dbname.empty()) { err = "no database selected"; return FlexData::make_object(); }
  auto keyOpt = b64_decode(keyb64);
  if (!keyOpt) { err = "bad key encoding"; return FlexData::make_object(); }
  const string key = *keyOpt;

  LmdbEnv* env = _sctx ? _sctx->lmdb_env() : nullptr;
  if (!env || !env->valid()) { err = "no database available"; return FlexData::make_object(); }
  MDB_env* e = env->raw();
  MDB_txn* txn = nullptr;
  if (mdb_txn_begin(e, nullptr, 0, &txn) != 0) {
    err = "could not open a write transaction"; return FlexData::make_object();
  }
  MDB_dbi dbi = 0;
  if (mdb_dbi_open(txn, dbname.c_str(), 0, &dbi) != 0) {
    mdb_txn_abort(txn);
    err = "no such database: " + dbname; return FlexData::make_object();
  }
  MDB_val mk;
  mk.mv_size = key.size();
  mk.mv_data = const_cast<char*>(key.data());
  int rc = mdb_del(txn, dbi, &mk, nullptr);
  if (rc != 0 && rc != MDB_NOTFOUND) {
    mdb_txn_abort(txn);
    err = "delete failed"; return FlexData::make_object();
  }
  if (mdb_txn_commit(txn) != 0) {
    err = "commit failed"; return FlexData::make_object();
  }

  FlexData doc = FlexData::make_object();
  auto d = doc.as_object();
  d.insert("ok", FlexData::make_bool(true));
  d.insert("deleted", FlexData::make_bool(rc == 0));
  return doc;
}

FlexData
DbBrowser::drop_database(const FlexData& req, string& err)
{
  if (!req.is_object()) { err = "expected a JSON object"; return FlexData::make_object(); }
  auto root = req.as_object();
  const string dbname = fstr_get(root, "db");
  if (dbname.empty()) { err = "no database selected"; return FlexData::make_object(); }

  LmdbEnv* env = _sctx ? _sctx->lmdb_env() : nullptr;
  if (!env || !env->valid()) { err = "no database available"; return FlexData::make_object(); }
  MDB_env* e = env->raw();
  MDB_txn* txn = nullptr;
  if (mdb_txn_begin(e, nullptr, 0, &txn) != 0) {
    err = "could not open a write transaction"; return FlexData::make_object();
  }
  MDB_dbi dbi = 0;
  if (mdb_dbi_open(txn, dbname.c_str(), 0, &dbi) != 0) {
    mdb_txn_abort(txn);
    err = "no such database: " + dbname; return FlexData::make_object();
  }
  // del=1 removes the named database from the environment (so it
  // vanishes from the list); del=0 would only empty it.
  if (mdb_drop(txn, dbi, 1) != 0) {
    mdb_txn_abort(txn);
    err = "drop failed"; return FlexData::make_object();
  }
  if (mdb_txn_commit(txn) != 0) {
    err = "commit failed"; return FlexData::make_object();
  }

  FlexData doc = FlexData::make_object();
  doc.as_object().insert("ok", FlexData::make_bool(true));
  return doc;
}

}
