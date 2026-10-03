#include "generative-models/train/sample-cache.h"

#include "common/flex-data.h"
#include "generative-models/train/lora-params.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <fcntl.h>
#include <unistd.h>

namespace vpipe {
namespace genai {
namespace train {

namespace fs = std::filesystem;

namespace {

constexpr int kIndexVersion = 1;

// The safetensors header write_safetensors() produces for `n` tensors of
// the given names, dtypes, shapes and sizes is not something to predict
// here: the offsets are taken from the file instead, by parsing its
// header back after the write.
bool
header_offsets_(const std::string& path,
                std::map<std::string, std::pair<std::uint64_t,
                                                std::uint64_t>>* out,
                std::string* err)
{
  std::ifstream in(path, std::ios::binary);
  std::uint64_t hn = 0;
  if (!in.read(reinterpret_cast<char*>(&hn), 8) || hn == 0 ||
      hn > (64ull << 20)) {
    if (err != nullptr) { *err = "cannot read the header of " + path; }
    return false;
  }
  std::string h(hn, '\0');
  if (!in.read(h.data(), (std::streamsize)hn)) {
    if (err != nullptr) { *err = "short header in " + path; }
    return false;
  }
  const FlexData j = FlexData::from_json(h);
  if (!j.is_object()) {
    if (err != nullptr) { *err = "bad header in " + path; }
    return false;
  }
  const std::uint64_t base = 8 + hn;
  auto o = j.as_object();
  for (const auto& kv : o) {
    const std::string name(kv.first);
    if (name == "__metadata__") { continue; }
    const FlexData& t = kv.second;
    auto to = t.as_object();
    if (!to.contains("data_offsets")) { continue; }
    const FlexData d = to.at("data_offsets");
    auto da = d.as_array();
    if (da.size() != 2) { continue; }
    (*out)[name] = {base + (std::uint64_t)da.at(0).as_uint(0),
                    (std::uint64_t)(da.at(1).as_uint(0) -
                                    da.at(0).as_uint(0))};
  }
  return true;
}

}  // namespace

std::string
cache_key(std::initializer_list<std::string_view> parts)
{
  std::uint64_t h = 1469598103934665603ull;
  auto mix = [&](unsigned char c) {
    h ^= c;
    h *= 1099511628211ull;
  };
  for (std::string_view p : parts) {
    for (char c : p) { mix((unsigned char)c); }
    mix(0x1f);   // a separator no part contains, so ("ab","c") != ("a","bc")
  }
  char buf[17];
  std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long)h);
  return buf;
}

SampleCache::~SampleCache()
{
  close();
}

void
SampleCache::close()
{
  for (int fd : _fds) {
    if (fd >= 0) { ::close(fd); }
  }
  _fds.clear();
  _dir.clear();
  _shards.clear();
  _entries.clear();
  _pending.clear();
  _pending_bytes = 0;
}

std::set<std::string>
SampleCache::keys_in(const std::string& dir)
{
  std::set<std::string> keys;
  std::ifstream in(fs::path(dir) / "index.json");
  if (!in) { return keys; }
  std::stringstream ss;
  ss << in.rdbuf();
  const FlexData j = FlexData::from_json(ss.str());
  if (!j.is_object()) { return keys; }
  auto o = j.as_object();
  if (!o.contains("version") || o.at("version").as_int(0) != kIndexVersion ||
      !o.contains("entries")) {
    return keys;
  }
  const FlexData e = o.at("entries");
  if (!e.is_object()) { return keys; }
  auto eo = e.as_object();
  for (const auto& kv : eo) { keys.insert(std::string(kv.first)); }
  return keys;
}

bool
SampleCache::open(const std::string& dir, std::string* err)
{
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) {
    if (err != nullptr) {
      *err = "cannot create the cache directory " + dir + ": " +
             ec.message();
    }
    return false;
  }
  close();
  _dir = dir;
  std::ifstream in(fs::path(dir) / "index.json");
  if (!in) { return true; }   // a new cache
  std::stringstream ss;
  ss << in.rdbuf();
  const FlexData j = FlexData::from_json(ss.str());
  if (!j.is_object()) { return true; }
  auto o = j.as_object();
  if (!o.contains("version") || o.at("version").as_int(0) != kIndexVersion) {
    return true;
  }
  if (o.contains("shards")) {
    const FlexData s = o.at("shards");
    auto sa = s.as_array();
    for (std::size_t i = 0; i < sa.size(); ++i) {
      _shards.push_back(std::string(sa.at(i).as_string("")));
    }
  }
  if (o.contains("entries")) {
    const FlexData e = o.at("entries");
    auto eo = e.as_object();
    for (const auto& kv : eo) {
      const FlexData& v = kv.second;
      auto vo = v.as_object();
      Entry en;
      en.shard = (int)vo.at("shard").as_int(-1);
      en.offset = vo.at("offset").as_uint(0);
      en.bytes = vo.at("bytes").as_uint(0);
      en.dtype = std::string(vo.at("dtype").as_string(""));
      const FlexData sh = vo.at("shape");
      auto sha = sh.as_array();
      for (std::size_t i = 0; i < sha.size(); ++i) {
        en.shape.push_back(sha.at(i).as_int(0));
      }
      if (vo.contains("sideband")) {
        en.sideband = std::string(vo.at("sideband").as_string(""));
      }
      if (en.shard < 0 || en.shard >= (int)_shards.size()) { continue; }
      _entries.emplace(std::string(kv.first), std::move(en));
    }
  }
  _fds.assign(_shards.size(), -1);
  return true;
}

bool
SampleCache::has(const std::string& key) const
{
  return _entries.count(key) != 0;
}

const SampleCache::Entry*
SampleCache::find(const std::string& key) const
{
  const auto it = _entries.find(key);
  return it == _entries.end() ? nullptr : &it->second;
}

void
SampleCache::put(const std::string& key, std::string dtype,
                 std::vector<std::int64_t> shape,
                 std::vector<std::uint8_t> bytes, std::string sideband)
{
  _pending_bytes += bytes.size();
  _pending.push_back(Pending{key, std::move(dtype), std::move(shape),
                             std::move(bytes), std::move(sideband)});
}

bool
SampleCache::flush(std::string* err)
{
  if (_pending.empty()) { return true; }
  if (_dir.empty()) {
    if (err != nullptr) { *err = "the sample cache is not open"; }
    return false;
  }
  char nm[32];
  std::snprintf(nm, sizeof nm, "shard-%05zu.safetensors", _shards.size());
  const std::string path = (fs::path(_dir) / nm).string();
  std::vector<OutTensor> ts;
  ts.reserve(_pending.size());
  for (Pending& p : _pending) {
    OutTensor t;
    t.name = p.key;
    t.dtype = p.dtype;
    t.shape = p.shape;
    t.bytes = std::move(p.bytes);
    ts.push_back(std::move(t));
  }
  if (!write_safetensors(path, ts, {{"format", "vpipe-train-cache"}}, err)) {
    return false;
  }
  std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> offs;
  if (!header_offsets_(path, &offs, err)) { return false; }
  const int shard = (int)_shards.size();
  _shards.push_back(nm);
  _fds.push_back(-1);
  for (Pending& p : _pending) {
    const auto it = offs.find(p.key);
    if (it == offs.end()) { continue; }
    Entry en;
    en.shard = shard;
    en.offset = it->second.first;
    en.bytes = it->second.second;
    en.dtype = p.dtype;
    en.shape = p.shape;
    en.sideband = std::move(p.sideband);
    _entries[p.key] = std::move(en);
  }
  _pending.clear();
  _pending_bytes = 0;
  return write_index_(err);
}

bool
SampleCache::write_index_(std::string* err) const
{
  FlexData j = FlexData::make_object();
  auto o = j.as_object();
  o.insert_or_assign("version", FlexData::make_int(kIndexVersion));
  FlexData sh = FlexData::make_array();
  for (const std::string& s : _shards) {
    sh.as_array().push_back(FlexData::make_string(s));
  }
  o.insert_or_assign("shards", std::move(sh));
  FlexData es = FlexData::make_object();
  auto eo = es.as_object();
  for (const auto& kv : _entries) {
    const Entry& en = kv.second;
    FlexData v = FlexData::make_object();
    auto vo = v.as_object();
    vo.insert_or_assign("shard", FlexData::make_int(en.shard));
    vo.insert_or_assign("offset", FlexData::make_uint(en.offset));
    vo.insert_or_assign("bytes", FlexData::make_uint(en.bytes));
    vo.insert_or_assign("dtype", FlexData::make_string(en.dtype));
    FlexData shp = FlexData::make_array();
    for (std::int64_t d : en.shape) {
      shp.as_array().push_back(FlexData::make_int(d));
    }
    vo.insert_or_assign("shape", std::move(shp));
    if (!en.sideband.empty()) {
      vo.insert_or_assign("sideband", FlexData::make_string(en.sideband));
    }
    eo.insert_or_assign(kv.first, std::move(v));
  }
  o.insert_or_assign("entries", std::move(es));
  const fs::path p = fs::path(_dir) / "index.json";
  const fs::path tmp = fs::path(_dir) / "index.json.tmp";
  {
    std::ofstream out(tmp, std::ios::trunc);
    out << j.to_json();
    if (!out) {
      if (err != nullptr) { *err = "cannot write " + tmp.string(); }
      return false;
    }
  }
  std::error_code ec;
  fs::rename(tmp, p, ec);
  if (ec) {
    if (err != nullptr) { *err = "rename to " + p.string() + ": " +
                                 ec.message(); }
    return false;
  }
  return true;
}

bool
SampleCache::read(const std::string& key, void* dst, std::size_t cap,
                  std::string* err) const
{
  const Entry* en = find(key);
  if (en == nullptr) {
    if (err != nullptr) { *err = "no cached entry " + key; }
    return false;
  }
  if (cap < en->bytes) {
    if (err != nullptr) { *err = "a destination too small for " + key; }
    return false;
  }
  int& fd = _fds[(std::size_t)en->shard];
  if (fd < 0) {
    const std::string path =
        (fs::path(_dir) / _shards[(std::size_t)en->shard]).string();
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      if (err != nullptr) { *err = "cannot open " + path; }
      return false;
    }
  }
  auto* d = static_cast<char*>(dst);
  std::uint64_t done = 0;
  while (done < en->bytes) {
    const ssize_t n = ::pread(fd, d + done, (std::size_t)(en->bytes - done),
                              (off_t)(en->offset + done));
    if (n <= 0) {
      if (err != nullptr) { *err = "short read of cached entry " + key; }
      return false;
    }
    done += (std::uint64_t)n;
  }
  return true;
}

}  // namespace train
}  // namespace genai
}  // namespace vpipe
