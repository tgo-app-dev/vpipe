#include "generative-models/safetensors-ranges.h"

#include "common/flex-data.h"
#include "generative-models/llama3/metal-llama-weights.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace vpipe::genai {

namespace fs = std::filesystem;

SafetensorsRanges::~SafetensorsRanges()
{
  for (int fd : _fds) {
    if (fd >= 0) { ::close(fd); }
  }
}

std::unique_ptr<SafetensorsRanges>
SafetensorsRanges::open(const std::string& dir, const MetalLlamaWeights& src,
                        std::string* err, bool cached)
{
  auto fail = [&](std::string m) -> std::unique_ptr<SafetensorsRanges> {
    if (err != nullptr) { *err = std::move(m); }
    return nullptr;
  };
  // name -> file: the index's weight map, or the one model.safetensors.
  std::unordered_map<std::string, std::string> file_of;
  {
    std::ifstream in(fs::path(dir) / "model.safetensors.index.json");
    if (in) {
      try {
        const FlexData idx = FlexData::from_json(in);
        const FlexData wm = idx.as_object().at("weight_map");
        for (const auto& [k, v] : wm.as_object()) {
          file_of.emplace(std::string(k), std::string(v.as_string("")));
        }
      } catch (...) {
        return fail("safetensors ranges: an unreadable index");
      }
    }
  }
  std::unique_ptr<SafetensorsRanges> r(new SafetensorsRanges());
  r->_cached = cached;
  std::unordered_map<std::string, int> fidx;
  std::vector<std::uint64_t> base;
  auto file_index = [&](const std::string& f) -> int {
    auto it = fidx.find(f);
    if (it != fidx.end()) { return it->second; }
    const std::string path = (fs::path(dir) / f).string();
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { return -1; }
    if (!cached) { (void)::fcntl(fd, F_NOCACHE, 1); }
    std::uint64_t hlen = 0;
    if (::pread(fd, &hlen, 8, 0) != 8) {
      ::close(fd);
      return -1;
    }
    r->_fds.push_back(fd);
    base.push_back(8 + hlen);
    const int i = (int)r->_fds.size() - 1;
    fidx.emplace(f, i);
    return i;
  };
  for (const std::string& nm : src.tensor_names()) {
    const auto* ti = src.info(nm);
    // A GGUF or Draw Things tensor is converted on the way out: no range.
    if (ti == nullptr || ti->shard < 0) { continue; }
    std::string f = "model.safetensors";
    if (!file_of.empty()) {
      auto it = file_of.find(nm);
      if (it == file_of.end()) { continue; }   // a tensor the index omits
      f = it->second;
    }
    const int i = file_index(f);
    if (i < 0) { return fail("safetensors ranges: cannot open " + f); }
    r->_where[nm] = {i, base[(std::size_t)i] + ti->offset, ti->nbytes};
  }
  return r;
}

std::size_t
SafetensorsRanges::nbytes(const std::string& name) const
{
  auto it = _where.find(name);
  return it == _where.end() ? 0 : (std::size_t)it->second.n;
}

bool
SafetensorsRanges::read(const std::string& name, std::size_t off,
                        std::size_t n, void* dst) const
{
  auto it = _where.find(name);
  if (it == _where.end() || off + n > it->second.n || dst == nullptr) {
    return false;
  }
  if (n == 0) { return true; }
  const int fd = _fds[(std::size_t)it->second.file];
  const std::uint64_t at = it->second.at + off;
  if (_cached) {
    auto* d = static_cast<std::uint8_t*>(dst);
    std::size_t got = 0;
    while (got < n) {
      const ssize_t k = ::pread(fd, d + got, n - got, (off_t)(at + got));
      if (k <= 0) { return false; }
      got += (std::size_t)k;
    }
    return true;
  }
  static const std::uint64_t page = (std::uint64_t)::sysconf(_SC_PAGESIZE);
  const std::uint64_t a0 = at / page * page;
  const std::uint64_t a1 = (at + n + page - 1) / page * page;
  const std::size_t span = (std::size_t)(a1 - a0);
  // One page-aligned staging buffer per thread, grown as needed.
  thread_local std::vector<std::uint8_t> stage;
  if (stage.size() < span + page) { stage.resize(span + page); }
  auto* buf = reinterpret_cast<std::uint8_t*>(
      ((std::uintptr_t)stage.data() + page - 1) / page * page);
  std::size_t got = 0;
  while (got < span) {
    const ssize_t k = ::pread(fd, buf + got, span - got, (off_t)(a0 + got));
    if (k <= 0) {
      // The file ends inside the last page: what was asked is all there.
      if (k == 0 && a0 + got >= at + n) { break; }
      return false;
    }
    got += (std::size_t)k;
  }
  if (a0 + got < at + n) { return false; }
  std::memcpy(dst, buf + (at - a0), n);
  return true;
}

bool
SafetensorsRanges::read_together(const std::vector<std::string>& names,
                                 std::vector<std::uint8_t>& stage,
                                 std::vector<const std::uint8_t*>& at) const
{
  if (names.empty()) { return false; }
  std::vector<const Where*> w;
  std::uint64_t lo = UINT64_MAX, hi = 0, sum = 0;
  for (const std::string& nm : names) {
    auto it = _where.find(nm);
    if (it == _where.end()) { return false; }
    if (!w.empty() && it->second.file != w[0]->file) { return false; }
    w.push_back(&it->second);
    lo = std::min(lo, it->second.at);
    hi = std::max(hi, it->second.at + it->second.n);
    sum += it->second.n;
  }
  if (hi - lo > sum + sum / 4) { return false; }
  static const std::uint64_t page = (std::uint64_t)::sysconf(_SC_PAGESIZE);
  const std::uint64_t a0 = lo / page * page;
  const std::uint64_t a1 = (hi + page - 1) / page * page;
  const std::size_t span = (std::size_t)(a1 - a0);
  if (stage.size() < span + page) { stage.resize(span + page); }
  auto* buf = reinterpret_cast<std::uint8_t*>(
      ((std::uintptr_t)stage.data() + page - 1) / page * page);
  const int fd = _fds[(std::size_t)w[0]->file];
  std::size_t got = 0;
  while (got < span) {
    const ssize_t k = ::pread(fd, buf + got, span - got, (off_t)(a0 + got));
    if (k <= 0) {
      if (k == 0 && a0 + got >= hi) { break; }
      return false;
    }
    got += (std::size_t)k;
  }
  if (a0 + got < hi) { return false; }
  at.resize(names.size());
  for (std::size_t i = 0; i < names.size(); ++i) {
    at[i] = buf + (w[i]->at - a0);
  }
  return true;
}

}  // namespace vpipe::genai
