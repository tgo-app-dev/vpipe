#include "generative-models/quantize/safetensors-writer.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <unistd.h>

namespace vpipe::genai {

namespace {

std::string
fmt_shard_name_(int idx, int total)
{
  char buf[64];
  std::snprintf(buf, sizeof(buf), "model-%05d-of-%05d.safetensors", idx + 1,
                total);
  return std::string(buf);
}

bool
write_all_(int fd, const void* p, std::size_t n)
{
  const auto* b = static_cast<const std::uint8_t*>(p);
  // macOS write() fails with EINVAL when the byte count exceeds INT_MAX, so
  // cap each call to 1 GiB. Untied embeddings / lm_head on large models (e.g.
  // the 27B's [248320, 5120] bf16 ~= 2.5 GB) are the first tensors big enough
  // to hit this; smaller tensors complete in a single iteration.
  constexpr std::size_t kChunk = std::size_t{1} << 30;   // 1 GiB
  while (n > 0) {
    const std::size_t want = n < kChunk ? n : kChunk;
    const ssize_t w = ::write(fd, b, want);
    if (w <= 0) { return false; }
    b += w;
    n -= static_cast<std::size_t>(w);
  }
  return true;
}

}  // namespace

SafetensorsWriter::SafetensorsWriter(std::string out_dir,
                                     std::uint64_t shard_max_bytes)
  : _out_dir(std::move(out_dir)), _shard_max(shard_max_bytes)
{
  std::error_code ec;
  std::filesystem::create_directories(_out_dir, ec);
}

SafetensorsWriter::~SafetensorsWriter()
{
  for (auto& sh : _shards) {
    if (sh.fd >= 0) { ::close(sh.fd); }
    if (!sh.tmp_path.empty()) {
      std::error_code ec;
      std::filesystem::remove(sh.tmp_path, ec);  // no-op if already renamed
    }
  }
}

bool
SafetensorsWriter::ensure_shard_()
{
  if (!_shards.empty()) {
    Shard& cur = _shards.back();
    if (cur.fd >= 0) { return true; }   // current shard still open
  }
  const int idx = static_cast<int>(_shards.size());
  Shard sh;
  sh.tmp_path =
      (std::filesystem::path(_out_dir) /
       (".shard-" + std::to_string(idx) + ".data.tmp")).string();
  sh.fd = ::open(sh.tmp_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (sh.fd < 0) { return false; }
  _shards.push_back(std::move(sh));
  return true;
}

bool
SafetensorsWriter::add(const std::string& name, const std::string& dtype,
                       const std::vector<std::int64_t>& shape,
                       const void* data, std::uint64_t nbytes)
{
  if (_closed) { return false; }
  // Roll to a new shard when the current one is non-empty and adding this
  // tensor would exceed the budget (a single oversized tensor still lands in
  // its own shard rather than being split).
  if (!_shards.empty() && _shards.back().fd >= 0 &&
      _shards.back().size > 0 && _shards.back().size + nbytes > _shard_max) {
    ::close(_shards.back().fd);
    _shards.back().fd = -1;
  }
  if (!ensure_shard_()) { return false; }
  Shard& cur = _shards.back();

  Entry e;
  e.name   = name;
  e.dtype  = dtype;
  e.shape  = shape;
  e.offset = cur.size;
  e.nbytes = nbytes;
  if (nbytes > 0 && !write_all_(cur.fd, data, nbytes)) { return false; }
  cur.size += nbytes;
  cur.entries.push_back(std::move(e));
  return true;
}

bool
SafetensorsWriter::finalize_shard_(int idx, int total, std::string& out_name)
{
  Shard& sh = _shards[static_cast<std::size_t>(idx)];

  // Build the JSON header (insertion order; the reader is order-agnostic).
  std::string hdr = "{";
  bool first = true;
  for (const auto& e : sh.entries) {
    if (!first) { hdr += ','; }
    first = false;
    hdr += '"';
    hdr += e.name;
    hdr += "\":{\"dtype\":\"";
    hdr += e.dtype;
    hdr += "\",\"shape\":[";
    for (std::size_t i = 0; i < e.shape.size(); ++i) {
      if (i) { hdr += ','; }
      hdr += std::to_string(e.shape[i]);
    }
    hdr += "],\"data_offsets\":[";
    hdr += std::to_string(e.offset);
    hdr += ',';
    hdr += std::to_string(e.offset + e.nbytes);
    hdr += "]}";
  }
  hdr += '}';

  out_name = fmt_shard_name_(idx, total);
  const std::string final_path =
      (std::filesystem::path(_out_dir) / out_name).string();
  const int out = ::open(final_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC,
                         0644);
  if (out < 0) { return false; }

  const std::uint64_t hlen = hdr.size();
  bool ok = write_all_(out, &hlen, 8) &&
            write_all_(out, hdr.data(), hdr.size());

  // Copy the streamed data blob from the temp file. The writer fd may
  // already be closed (a rolled shard closes its fd in add()), so reopen
  // by path for reading rather than relying on sh.fd.
  if (sh.fd >= 0) { ::close(sh.fd); sh.fd = -1; }
  if (ok && sh.size > 0) {
    const int in = ::open(sh.tmp_path.c_str(), O_RDONLY);
    if (in < 0) {
      ok = false;
    } else {
      std::vector<std::uint8_t> buf(8u << 20);   // 8 MB
      std::uint64_t left = sh.size;
      while (ok && left > 0) {
        const std::size_t want = static_cast<std::size_t>(
            std::min<std::uint64_t>(left, buf.size()));
        const ssize_t r = ::read(in, buf.data(), want);
        if (r <= 0) { ok = false; break; }
        ok = write_all_(out, buf.data(), static_cast<std::size_t>(r));
        left -= static_cast<std::uint64_t>(r);
      }
      ::close(in);
    }
  }
  ::close(out);
  std::error_code ec;
  std::filesystem::remove(sh.tmp_path, ec);
  sh.tmp_path.clear();
  return ok;
}

bool
SafetensorsWriter::close()
{
  if (_closed) { return true; }
  _closed = true;
  if (_shards.empty()) { return true; }

  const int total = static_cast<int>(_shards.size());
  std::string idx_json = "{\"metadata\":{\"total_size\":0},\"weight_map\":{";
  bool first = true;
  for (int i = 0; i < total; ++i) {
    std::string shard_name;
    if (!finalize_shard_(i, total, shard_name)) { return false; }
    for (const auto& e : _shards[static_cast<std::size_t>(i)].entries) {
      if (!first) { idx_json += ','; }
      first = false;
      idx_json += '"';
      idx_json += e.name;
      idx_json += "\":\"";
      idx_json += shard_name;
      idx_json += '"';
    }
  }
  idx_json += "}}";

  const std::string idx_path =
      (std::filesystem::path(_out_dir) / "model.safetensors.index.json")
          .string();
  const int fd = ::open(idx_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) { return false; }
  const bool ok = write_all_(fd, idx_json.data(), idx_json.size());
  ::close(fd);
  return ok;
}

}  // namespace vpipe::genai
