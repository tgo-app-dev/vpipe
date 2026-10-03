#include "generative-models/train/lora-params.h"

#include "generative-models/llama3/metal-llama-weights.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>

namespace vpipe {
namespace genai {
namespace train {

using metal_compute::SharedBuffer;

std::uint16_t
f32_to_bf16(float f) noexcept
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (std::uint16_t)(u >> 16);
}

float
bf16_to_f32(std::uint16_t h) noexcept
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

namespace {

float
f16_to_f32_(std::uint16_t h)
{
  const std::uint32_t s = (std::uint32_t)(h >> 15) << 31;
  std::uint32_t e = (h >> 10) & 0x1f;
  std::uint32_t m = h & 0x3ff;
  std::uint32_t u;
  if (e == 0) {
    if (m == 0) {
      u = s;
    } else {
      e = 127 - 15 + 1;
      while ((m & 0x400) == 0) { m <<= 1; --e; }
      m &= 0x3ff;
      u = s | (e << 23) | (m << 13);
    }
  } else if (e == 31) {
    u = s | 0x7f800000u | (m << 13);
  } else {
    u = s | ((e + 127 - 15) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::string
json_escape_(const std::string& s)
{
  std::string o;
  o.reserve(s.size() + 2);
  for (char c : s) {
    switch (c) {
    case '"': o += "\\\""; break;
    case '\\': o += "\\\\"; break;
    case '\n': o += "\\n"; break;
    case '\t': o += "\\t"; break;
    case '\r': o += "\\r"; break;
    default:
      if ((unsigned char)c < 0x20) {
        char b[8];
        std::snprintf(b, sizeof b, "\\u%04x", (unsigned)c);
        o += b;
      } else {
        o += c;
      }
    }
  }
  return o;
}

// The A shadow as the runtime loader makes it from a bf16 file.
inline std::uint16_t
shadow_(float master, float mul)
{
  return f32_to_bf16(bf16_to_f32(f32_to_bf16(master)) * mul);
}

}  // namespace

bool
write_safetensors(const std::string& path,
                  const std::vector<OutTensor>& tensors,
                  const std::map<std::string, std::string>& meta,
                  std::string* err)
{
  std::string h = "{";
  bool first = true;
  if (!meta.empty()) {
    h += "\"__metadata__\":{";
    bool f2 = true;
    for (const auto& kv : meta) {
      if (!f2) { h += ","; }
      f2 = false;
      h += "\"" + json_escape_(kv.first) + "\":\"" +
           json_escape_(kv.second) + "\"";
    }
    h += "}";
    first = false;
  }
  std::uint64_t off = 0;
  for (const OutTensor& t : tensors) {
    if (!first) { h += ","; }
    first = false;
    h += "\"" + json_escape_(t.name) + "\":{\"dtype\":\"" + t.dtype +
         "\",\"shape\":[";
    for (std::size_t i = 0; i < t.shape.size(); ++i) {
      if (i > 0) { h += ","; }
      h += std::to_string(t.shape[i]);
    }
    h += "],\"data_offsets\":[" + std::to_string(off) + "," +
         std::to_string(off + t.bytes.size()) + "]}";
    off += t.bytes.size();
  }
  h += "}";
  // Pad the header with spaces so the data blob starts 8-aligned, as the
  // reference writer does.
  while ((h.size() % 8) != 0) { h += ' '; }

  const std::string tmp = path + ".tmp";
  {
    std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
    if (!o) {
      if (err != nullptr) { *err = "cannot write " + tmp; }
      return false;
    }
    const std::uint64_t hn = h.size();
    o.write(reinterpret_cast<const char*>(&hn), 8);
    o.write(h.data(), (std::streamsize)h.size());
    for (const OutTensor& t : tensors) {
      o.write(reinterpret_cast<const char*>(t.bytes.data()),
              (std::streamsize)t.bytes.size());
    }
    if (!o) {
      if (err != nullptr) { *err = "short write to " + tmp; }
      return false;
    }
  }
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    if (err != nullptr) { *err = "rename to " + path + ": " + ec.message(); }
    return false;
  }
  return true;
}

bool
LoraParams::create(metal_compute::MetalCompute* mc,
                   const std::vector<ModuleShape>& mods, int rank,
                   float alpha, std::string* err)
{
  _mc = mc;
  _rank = rank;
  _alpha = alpha;
  _mods.clear();
  _na = _nb = 0;
  if (mc == nullptr || rank <= 0 || mods.empty()) {
    if (err != nullptr) { *err = "lora params: nothing to train"; }
    return false;
  }
  for (const ModuleShape& m : mods) {
    Module x;
    x.name = m.name;
    x.n = m.n;
    x.k = m.k;
    x.a_off = _na;
    x.b_off = _nb;
    _na += (std::size_t)rank * m.k;
    _nb += (std::size_t)m.n * rank;
    _mods.push_back(std::move(x));
  }
  auto make = [&](std::size_t elems, std::size_t esz) {
    SharedBuffer b = mc->make_shared_buffer(elems * esz);
    if (!b.empty()) { std::memset(b.contents(), 0, b.byte_size()); }
    return b;
  };
  _a_m = make(_na, 4); _b_m = make(_nb, 4);
  _a_g = make(_na, 4); _b_g = make(_nb, 4);
  _a_s = make(_na, 2); _b_s = make(_nb, 2);
  if (_a_m.empty() || _b_m.empty() || _a_g.empty() || _b_g.empty() ||
      _a_s.empty() || _b_s.empty()) {
    if (err != nullptr) { *err = "lora params: arena allocation failed"; }
    return false;
  }
  for (Module& m : _mods) {
    m.live.rank = rank;
    m.live.a = _a_s.subview(m.a_off * 2, (std::size_t)rank * m.k * 2);
    m.live.b = _b_s.subview(m.b_off * 2, (std::size_t)m.n * rank * 2);
  }
  return true;
}

void
LoraParams::init_random(std::uint64_t seed)
{
  std::mt19937_64 g(seed);
  auto* a = static_cast<float*>(_a_m.contents());
  auto* b = static_cast<float*>(_b_m.contents());
  for (const Module& m : _mods) {
    // PEFT's init: kaiming_uniform_(a=sqrt(5)), whose bound over a fan-in
    // of k is 1 / sqrt(k).
    const float bound = 1.0f / std::sqrt((float)m.k);
    std::uniform_real_distribution<float> d(-bound, bound);
    for (std::size_t i = 0; i < (std::size_t)_rank * m.k; ++i) {
      a[m.a_off + i] = d(g);
    }
    for (std::size_t i = 0; i < (std::size_t)m.n * _rank; ++i) {
      b[m.b_off + i] = 0.0f;
    }
  }
  refresh_shadows_cpu();
}

void
LoraParams::refresh_shadows_cpu()
{
  const auto* a = static_cast<const float*>(_a_m.contents());
  const auto* b = static_cast<const float*>(_b_m.contents());
  auto* as = static_cast<std::uint16_t*>(_a_s.contents());
  auto* bs = static_cast<std::uint16_t*>(_b_s.contents());
  const float mul = a_mul();
  for (std::size_t i = 0; i < _na; ++i) { as[i] = shadow_(a[i], mul); }
  for (std::size_t i = 0; i < _nb; ++i) { bs[i] = shadow_(b[i], 1.0f); }
}

const LoraParams::Module*
LoraParams::find(const std::string& name) const
{
  for (const Module& m : _mods) {
    if (m.name == name) { return &m; }
  }
  return nullptr;
}

bool
LoraParams::load(const std::string& path, int* loaded, std::string* err)
{
  if (loaded != nullptr) { *loaded = 0; }
  auto w = MetalLlamaWeights::open(path);
  if (!w.has_value()) {
    if (err != nullptr) { *err = "cannot open " + path; }
    return false;
  }
  // The file's own alpha: a factor trained at alpha_f is rescaled into
  // this set's alpha, so the adapter's product -- what it DOES -- is kept.
  float file_alpha = 0.0f;
  for (const char* key : {"alpha", "lora_alpha"}) {
    auto it = w->metadata().find(key);
    if (it != w->metadata().end()) {
      file_alpha = std::strtof(it->second.c_str(), nullptr);
      if (file_alpha > 0.0f) { break; }
    }
  }
  auto read = [&](const std::string& module, const char* ab,
                  std::vector<float>* out, std::vector<std::int64_t>* shape) {
    for (const char* pre : {"transformer.", "", "diffusion_model."}) {
      for (const char* inf : {".weight", ".default.weight"}) {
        const std::string key =
            std::string(pre) + module + ".lora_" + ab + inf;
        const auto* info = w->info(key);
        if (info == nullptr) { continue; }
        std::size_t n = 1;
        for (auto d : info->shape) { n *= (std::size_t)d; }
        std::vector<std::uint8_t> raw(info->nbytes);
        if (!w->read_into(key, raw.data(), raw.size())) { return false; }
        out->resize(n);
        if (info->dtype == "F32" && raw.size() == n * 4) {
          std::memcpy(out->data(), raw.data(), n * 4);
        } else if (info->dtype == "BF16" && raw.size() == n * 2) {
          const auto* s = reinterpret_cast<const std::uint16_t*>(raw.data());
          for (std::size_t i = 0; i < n; ++i) { (*out)[i] = bf16_to_f32(s[i]); }
        } else if (info->dtype == "F16" && raw.size() == n * 2) {
          const auto* s = reinterpret_cast<const std::uint16_t*>(raw.data());
          for (std::size_t i = 0; i < n; ++i) { (*out)[i] = f16_to_f32_(s[i]); }
        } else {
          return false;
        }
        *shape = info->shape;
        return true;
      }
    }
    return false;
  };
  const float rescale =
      file_alpha > 0.0f ? file_alpha / _alpha : (float)_rank / _alpha;
  auto* am = static_cast<float*>(_a_m.contents());
  auto* bm = static_cast<float*>(_b_m.contents());
  int n = 0;
  for (Module& m : _mods) {
    std::vector<float> a, b;
    std::vector<std::int64_t> sa, sb;
    const bool ha = read(m.name, "A", &a, &sa);
    const bool hb = read(m.name, "B", &b, &sb);
    if (!ha && !hb) { continue; }
    if (!ha || !hb || sa.size() != 2 || sb.size() != 2 ||
        sa[0] != _rank || sa[1] != m.k || sb[0] != m.n || sb[1] != _rank) {
      if (err != nullptr) {
        *err = "adapter module " + m.name + " does not match rank " +
               std::to_string(_rank) + " and this model's shape";
      }
      return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
      am[m.a_off + i] = a[i] * rescale;
    }
    std::memcpy(bm + m.b_off, b.data(), b.size() * 4);
    ++n;
  }
  if (loaded != nullptr) { *loaded = n; }
  refresh_shadows_cpu();
  return true;
}

bool
LoraParams::save(const std::string& path,
                 const std::map<std::string, std::string>& meta,
                 std::string* err, const SharedBuffer* ema_a,
                 const SharedBuffer* ema_b) const
{
  const auto* a = static_cast<const float*>(
      (ema_a != nullptr ? *ema_a : _a_m).contents());
  const auto* b = static_cast<const float*>(
      (ema_b != nullptr ? *ema_b : _b_m).contents());
  std::vector<OutTensor> out;
  out.reserve(_mods.size() * 2);
  for (const Module& m : _mods) {
    OutTensor ta, tb;
    ta.name = "transformer." + m.name + ".lora_A.weight";
    tb.name = "transformer." + m.name + ".lora_B.weight";
    ta.dtype = tb.dtype = "BF16";
    ta.shape = {_rank, m.k};
    tb.shape = {m.n, _rank};
    ta.bytes.resize((std::size_t)_rank * m.k * 2);
    tb.bytes.resize((std::size_t)m.n * _rank * 2);
    auto* da = reinterpret_cast<std::uint16_t*>(ta.bytes.data());
    auto* db = reinterpret_cast<std::uint16_t*>(tb.bytes.data());
    for (std::size_t i = 0; i < (std::size_t)_rank * m.k; ++i) {
      da[i] = f32_to_bf16(a[m.a_off + i]);
    }
    for (std::size_t i = 0; i < (std::size_t)m.n * _rank; ++i) {
      db[i] = f32_to_bf16(b[m.b_off + i]);
    }
    out.push_back(std::move(ta));
    out.push_back(std::move(tb));
  }
  std::map<std::string, std::string> md = meta;
  char buf[64];
  std::snprintf(buf, sizeof buf, "%g", (double)_alpha);
  md["alpha"] = buf;
  md["rank"] = std::to_string(_rank);
  return write_safetensors(path, out, md, err);
}

}  // namespace train
}  // namespace genai
}  // namespace vpipe
