#include "generative-models/shared/gguf-file.h"

#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vpipe::genai {

namespace {

// gguf_metadata_value_type
enum : std::uint32_t {
  kU8 = 0, kI8 = 1, kU16 = 2, kI16 = 3, kU32 = 4, kI32 = 5, kF32m = 6,
  kBoolm = 7, kStringm = 8, kArray = 9, kU64 = 10, kI64 = 11, kF64m = 12,
};

std::size_t
scalar_size_(std::uint32_t t)
{
  switch (t) {
    case kU8: case kI8: case kBoolm: return 1;
    case kU16: case kI16: return 2;
    case kU32: case kI32: case kF32m: return 4;
    case kU64: case kI64: case kF64m: return 8;
    default: return 0;   // string/array handled separately
  }
}

inline float
f16_to_f32_(std::uint16_t h)
{
  _Float16 v;
  std::memcpy(&v, &h, 2);
  return static_cast<float>(v);
}
inline float
bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// ggml block geometry for the types we size/dequant.
struct BlkGeom { int elems; int bytes; };
BlkGeom
blk_geom_(std::uint32_t type)
{
  switch (type) {
    case GgufFile::kF32:  return {1, 4};
    case GgufFile::kF16:  return {1, 2};
    case GgufFile::kBF16: return {1, 2};
    case GgufFile::kQ4_0: return {32, 18};
    case GgufFile::kQ8_0: return {32, 34};
    case GgufFile::kQ4_K: return {256, 144};
    case GgufFile::kQ5_K: return {256, 176};
    case GgufFile::kQ6_K: return {256, 210};
    default: return {0, 0};
  }
}

// A cursor over the mmap that the header/KV/tensor parse advances.
struct Reader {
  const std::uint8_t* base;
  std::size_t         size;
  std::uint64_t       pos = 0;
  bool                ok = true;

  bool need(std::uint64_t n) {
    if (!ok || pos + n > size) { ok = false; return false; }
    return true;
  }
  std::uint8_t  u8()  { if (!need(1)) return 0; return base[pos++]; }
  std::uint16_t u16() { if (!need(2)) return 0;
    std::uint16_t v; std::memcpy(&v, base + pos, 2); pos += 2; return v; }
  std::uint32_t u32() { if (!need(4)) return 0;
    std::uint32_t v; std::memcpy(&v, base + pos, 4); pos += 4; return v; }
  std::uint64_t u64() { if (!need(8)) return 0;
    std::uint64_t v; std::memcpy(&v, base + pos, 8); pos += 8; return v; }
  std::string str() {
    std::uint64_t n = u64();
    if (!need(n)) { return {}; }
    std::string s(reinterpret_cast<const char*>(base + pos),
                  static_cast<std::size_t>(n));
    pos += n;
    return s;
  }
  // Skip a metadata value of type `t`, recording array element offset
  // into *arr_off / *arr_elem / *arr_count when it is an array.
  void skip_value(std::uint32_t t, std::uint32_t* arr_elem,
                  std::uint64_t* arr_count, std::uint64_t* arr_off) {
    if (t == kStringm) { str(); return; }
    if (t == kArray) {
      std::uint32_t et = u32();
      std::uint64_t cnt = u64();
      if (arr_elem) { *arr_elem = et; }
      if (arr_count) { *arr_count = cnt; }
      if (arr_off) { *arr_off = pos; }
      if (et == kStringm) {
        for (std::uint64_t i = 0; i < cnt && ok; ++i) { str(); }
      } else {
        std::size_t es = scalar_size_(et);
        if (es == 0) { ok = false; return; }
        if (!need(cnt * es)) { return; }
        pos += cnt * es;
      }
      return;
    }
    std::size_t s = scalar_size_(t);
    if (s == 0) { ok = false; return; }
    if (!need(s)) { return; }
    pos += s;
  }
};

}  // namespace

std::int64_t
GgufFile::Tensor::numel() const
{
  std::int64_t n = 1;
  for (auto d : dims) { n *= d; }
  return n;
}

std::optional<GgufFile>
GgufFile::open(const std::string& path)
{
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) { return std::nullopt; }
  struct stat st {};
  if (::fstat(fd, &st) != 0 || st.st_size < 24) {
    ::close(fd);
    return std::nullopt;
  }
  const std::size_t fsz = static_cast<std::size_t>(st.st_size);
  void* base = ::mmap(nullptr, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
  if (base == MAP_FAILED) {
    ::close(fd);
    return std::nullopt;
  }

  GgufFile g;
  g._fd = fd;
  g._base = static_cast<const std::uint8_t*>(base);
  g._map_size = fsz;

  Reader r{ g._base, fsz, 0, true };
  if (r.u8() != 'G' || r.u8() != 'G' || r.u8() != 'U' || r.u8() != 'F') {
    return std::nullopt;   // moves out g -> munmap in dtor
  }
  const std::uint32_t version = r.u32();
  if (version != 2 && version != 3) { return std::nullopt; }
  const std::uint64_t n_tensors = r.u64();
  const std::uint64_t n_kv = r.u64();

  std::uint32_t alignment = 32;
  for (std::uint64_t i = 0; i < n_kv && r.ok; ++i) {
    std::string key = r.str();
    std::uint32_t vt = r.u32();
    Value v;
    v.type = vt;
    if (vt == kStringm) {
      v.str = r.str();
    } else if (vt == kArray) {
      r.skip_value(vt, &v.arr_elem, &v.arr_count, &v.arr_off);
    } else {
      std::size_t s = scalar_size_(vt);
      if (s == 0 || !r.need(s)) { return std::nullopt; }
      std::uint64_t bits = 0;
      std::memcpy(&bits, r.base + r.pos, s);
      r.pos += s;
      v.scalar = bits;
    }
    if (!r.ok) { return std::nullopt; }
    if (key == "general.alignment" && vt == kU32) {
      alignment = static_cast<std::uint32_t>(v.scalar);
    }
    g._meta.emplace(std::move(key), std::move(v));
  }
  if (!r.ok) { return std::nullopt; }

  g._tensors.reserve(static_cast<std::size_t>(n_tensors));
  std::vector<std::pair<std::uint32_t, std::uint64_t>> typ_off;
  typ_off.reserve(static_cast<std::size_t>(n_tensors));
  for (std::uint64_t i = 0; i < n_tensors && r.ok; ++i) {
    Tensor t;
    t.name = r.str();
    std::uint32_t nd = r.u32();
    if (nd > 8) { return std::nullopt; }
    t.dims.resize(nd);
    for (std::uint32_t d = 0; d < nd; ++d) {
      t.dims[d] = static_cast<std::int64_t>(r.u64());
    }
    t.type = r.u32();
    t.offset = r.u64();
    typ_off.emplace_back(t.type, t.offset);
    g._tensor_index.emplace(t.name, g._tensors.size());
    g._tensors.push_back(std::move(t));
  }
  if (!r.ok) { return std::nullopt; }

  if (alignment == 0) { alignment = 32; }
  g._data_start = (r.pos + alignment - 1) / alignment * alignment;

  for (auto& t : g._tensors) {
    if (g._data_start + t.offset > fsz) { return std::nullopt; }
    t.data = g._base + g._data_start + t.offset;
    BlkGeom bg = blk_geom_(t.type);
    if (bg.elems > 0) {
      const std::int64_t ne = t.numel();
      t.nbytes = static_cast<std::uint64_t>(ne / bg.elems) * bg.bytes;
    }
  }
  return g;
}

GgufFile::GgufFile(GgufFile&& o) noexcept { *this = std::move(o); }

GgufFile&
GgufFile::operator=(GgufFile&& o) noexcept
{
  if (this != &o) {
    if (_base != nullptr) { ::munmap(const_cast<std::uint8_t*>(_base),
                                     _map_size); }
    if (_fd >= 0) { ::close(_fd); }
    _fd = o._fd; _base = o._base; _map_size = o._map_size;
    _data_start = o._data_start;
    _meta = std::move(o._meta);
    _tensors = std::move(o._tensors);
    _tensor_index = std::move(o._tensor_index);
    o._fd = -1; o._base = nullptr; o._map_size = 0;
  }
  return *this;
}

GgufFile::~GgufFile()
{
  if (_base != nullptr) { ::munmap(const_cast<std::uint8_t*>(_base),
                                   _map_size); }
  if (_fd >= 0) { ::close(_fd); }
}

const GgufFile::Value*
GgufFile::find_(const std::string& key) const
{
  auto it = _meta.find(key);
  return it == _meta.end() ? nullptr : &it->second;
}

bool
GgufFile::has(const std::string& key) const
{
  return _meta.find(key) != _meta.end();
}

std::optional<std::uint64_t>
GgufFile::get_uint(const std::string& key) const
{
  const Value* v = find_(key);
  if (v == nullptr) { return std::nullopt; }
  switch (v->type) {
    case kU8: case kU16: case kU32: case kU64: case kBoolm:
      return v->scalar;
    case kI8:  return static_cast<std::uint64_t>(
                   static_cast<std::int64_t>(static_cast<std::int8_t>(
                       v->scalar)));
    case kI16: return static_cast<std::uint64_t>(
                   static_cast<std::int64_t>(static_cast<std::int16_t>(
                       v->scalar)));
    case kI32: return static_cast<std::uint64_t>(
                   static_cast<std::int64_t>(static_cast<std::int32_t>(
                       v->scalar)));
    case kI64: return v->scalar;
    default: return std::nullopt;
  }
}

std::optional<std::int64_t>
GgufFile::get_int(const std::string& key) const
{
  auto u = get_uint(key);
  if (!u) { return std::nullopt; }
  return static_cast<std::int64_t>(*u);
}

std::optional<double>
GgufFile::get_float(const std::string& key) const
{
  const Value* v = find_(key);
  if (v == nullptr) { return std::nullopt; }
  if (v->type == kF32m) {
    std::uint32_t b = static_cast<std::uint32_t>(v->scalar);
    float f; std::memcpy(&f, &b, 4); return f;
  }
  if (v->type == kF64m) {
    double d; std::memcpy(&d, &v->scalar, 8); return d;
  }
  auto i = get_int(key);
  if (i) { return static_cast<double>(*i); }
  return std::nullopt;
}

std::optional<bool>
GgufFile::get_bool(const std::string& key) const
{
  const Value* v = find_(key);
  if (v == nullptr || v->type != kBoolm) { return std::nullopt; }
  return v->scalar != 0;
}

std::optional<std::string>
GgufFile::get_string(const std::string& key) const
{
  const Value* v = find_(key);
  if (v == nullptr || v->type != kStringm) { return std::nullopt; }
  return v->str;
}

std::size_t
GgufFile::array_len(const std::string& key) const
{
  const Value* v = find_(key);
  if (v == nullptr || v->type != kArray) { return 0; }
  return static_cast<std::size_t>(v->arr_count);
}

std::vector<std::int64_t>
GgufFile::get_int_array(const std::string& key) const
{
  std::vector<std::int64_t> out;
  const Value* v = find_(key);
  if (v == nullptr || v->type != kArray) { return out; }
  const std::size_t es = scalar_size_(v->arr_elem);
  if (es == 0) { return out; }
  out.reserve(static_cast<std::size_t>(v->arr_count));
  std::uint64_t p = v->arr_off;
  for (std::uint64_t i = 0; i < v->arr_count; ++i, p += es) {
    if (p + es > _map_size) { break; }
    std::uint64_t bits = 0;
    std::memcpy(&bits, _base + p, es);
    std::int64_t val = 0;
    switch (v->arr_elem) {
      case kU8:  val = static_cast<std::uint8_t>(bits); break;
      case kI8:  val = static_cast<std::int8_t>(bits); break;
      case kU16: val = static_cast<std::uint16_t>(bits); break;
      case kI16: val = static_cast<std::int16_t>(bits); break;
      case kU32: val = static_cast<std::uint32_t>(bits); break;
      case kI32: val = static_cast<std::int32_t>(bits); break;
      case kU64: case kI64: val = static_cast<std::int64_t>(bits); break;
      case kBoolm: val = (bits != 0); break;
      default: break;
    }
    out.push_back(val);
  }
  return out;
}

std::vector<float>
GgufFile::get_float_array(const std::string& key) const
{
  std::vector<float> out;
  const Value* v = find_(key);
  if (v == nullptr || v->type != kArray) { return out; }
  out.reserve(static_cast<std::size_t>(v->arr_count));
  std::uint64_t p = v->arr_off;
  if (v->arr_elem == kF32m) {
    for (std::uint64_t i = 0; i < v->arr_count; ++i, p += 4) {
      if (p + 4 > _map_size) { break; }
      float f; std::memcpy(&f, _base + p, 4); out.push_back(f);
    }
  } else if (v->arr_elem == kF64m) {
    for (std::uint64_t i = 0; i < v->arr_count; ++i, p += 8) {
      if (p + 8 > _map_size) { break; }
      double d; std::memcpy(&d, _base + p, 8);
      out.push_back(static_cast<float>(d));
    }
  }
  return out;
}

std::vector<std::string>
GgufFile::get_string_array(const std::string& key) const
{
  std::vector<std::string> out;
  const Value* v = find_(key);
  if (v == nullptr || v->type != kArray || v->arr_elem != kStringm) {
    return out;
  }
  out.reserve(static_cast<std::size_t>(v->arr_count));
  std::uint64_t p = v->arr_off;
  for (std::uint64_t i = 0; i < v->arr_count; ++i) {
    if (p + 8 > _map_size) { break; }
    std::uint64_t n;
    std::memcpy(&n, _base + p, 8);
    p += 8;
    if (p + n > _map_size) { break; }
    out.emplace_back(reinterpret_cast<const char*>(_base + p),
                     static_cast<std::size_t>(n));
    p += n;
  }
  return out;
}

const GgufFile::Tensor*
GgufFile::tensor(const std::string& name) const
{
  auto it = _tensor_index.find(name);
  return it == _tensor_index.end() ? nullptr : &_tensors[it->second];
}

namespace {
// Unpack the 6-bit scale + 6-bit min for sub-block `j` (0..7) from a
// Q4_K/Q5_K super-block's 12-byte `scales` field. Bit-identical to
// llama.cpp's get_scale_min_k4.
inline void
get_scale_min_k4_(int j, const std::uint8_t* q, std::uint8_t* d,
                  std::uint8_t* m)
{
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
  }
}

// Dequantise `n` contiguous elements (block-aligned for the K/Q types)
// starting at `src` into `out` (n floats).
bool
dequant_span_(std::uint32_t type, const std::uint8_t* src,
              std::int64_t n, float* out)
{
  switch (type) {
    case GgufFile::kF32:
      std::memcpy(out, src, static_cast<std::size_t>(n) * 4);
      return true;
    case GgufFile::kF16: {
      const auto* s = reinterpret_cast<const std::uint16_t*>(src);
      for (std::int64_t i = 0; i < n; ++i) { out[i] = f16_to_f32_(s[i]); }
      return true;
    }
    case GgufFile::kBF16: {
      const auto* s = reinterpret_cast<const std::uint16_t*>(src);
      for (std::int64_t i = 0; i < n; ++i) { out[i] = bf16_to_f32_(s[i]); }
      return true;
    }
    case GgufFile::kQ8_0: {
      const std::int64_t nb = n / 32;
      const std::uint8_t* p = src;
      for (std::int64_t b = 0; b < nb; ++b, p += 34) {
        std::uint16_t d16; std::memcpy(&d16, p, 2);
        const float d = f16_to_f32_(d16);
        const auto* q = reinterpret_cast<const std::int8_t*>(p + 2);
        for (int j = 0; j < 32; ++j) { out[b * 32 + j] = d * q[j]; }
      }
      return true;
    }
    case GgufFile::kQ4_0: {
      const std::int64_t nb = n / 32;
      const std::uint8_t* p = src;
      for (std::int64_t b = 0; b < nb; ++b, p += 18) {
        std::uint16_t d16; std::memcpy(&d16, p, 2);
        const float d = f16_to_f32_(d16);
        const std::uint8_t* qs = p + 2;
        for (int j = 0; j < 16; ++j) {
          out[b * 32 + j]      = ((qs[j] & 0x0F) - 8) * d;
          out[b * 32 + j + 16] = ((qs[j] >> 4)  - 8) * d;
        }
      }
      return true;
    }
    case GgufFile::kQ4_K: {
      // 256-weight super-block in 144 bytes: d(f16) dmin(f16) scales[12]
      // qs[128]. Eight 32-weight sub-blocks; sub-block scale/min from the
      // 6-bit packed `scales`. Bit-identical to llama.cpp dequant_row_q4_K.
      const std::int64_t nsb = n / 256;
      const std::uint8_t* p = src;
      for (std::int64_t sb = 0; sb < nsb; ++sb, p += 144) {
        std::uint16_t d16, m16;
        std::memcpy(&d16, p, 2);
        std::memcpy(&m16, p + 2, 2);
        const float d = f16_to_f32_(d16);
        const float dmin = f16_to_f32_(m16);
        const std::uint8_t* scales = p + 4;
        const std::uint8_t* qs = p + 16;
        float* y = out + sb * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
          std::uint8_t sc, m;
          get_scale_min_k4_(is + 0, scales, &sc, &m);
          const float d1 = d * sc, m1 = dmin * m;
          get_scale_min_k4_(is + 1, scales, &sc, &m);
          const float d2 = d * sc, m2 = dmin * m;
          const std::uint8_t* q = qs + (j / 64) * 32;
          for (int l = 0; l < 32; ++l) {
            y[j + l]      = d1 * (q[l] & 0x0F) - m1;
            y[j + l + 32] = d2 * (q[l] >> 4)  - m2;
          }
          is += 2;
        }
      }
      return true;
    }
    case GgufFile::kQ5_K: {
      // 256-weight super-block in 176 bytes: d(f16) dmin(f16) scales[12]
      // qh[32] qs[128]. Like Q4_K plus a 5th bit per weight from qh.
      // Bit-identical to llama.cpp dequant_row_q5_K.
      const std::int64_t nsb = n / 256;
      const std::uint8_t* p = src;
      for (std::int64_t sb = 0; sb < nsb; ++sb, p += 176) {
        std::uint16_t d16, m16;
        std::memcpy(&d16, p, 2);
        std::memcpy(&m16, p + 2, 2);
        const float d = f16_to_f32_(d16);
        const float dmin = f16_to_f32_(m16);
        const std::uint8_t* scales = p + 4;
        const std::uint8_t* qh = p + 16;
        const std::uint8_t* qs = p + 48;
        float* y = out + sb * 256;
        int is = 0;
        std::uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
          std::uint8_t sc, m;
          get_scale_min_k4_(is + 0, scales, &sc, &m);
          const float d1 = d * sc, m1 = dmin * m;
          get_scale_min_k4_(is + 1, scales, &sc, &m);
          const float d2 = d * sc, m2 = dmin * m;
          const std::uint8_t* q = qs + (j / 64) * 32;
          for (int l = 0; l < 32; ++l) {
            const int lo = (q[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
            const int hi = (q[l] >> 4)   + ((qh[l] & u2) ? 16 : 0);
            y[j + l]      = d1 * lo - m1;
            y[j + l + 32] = d2 * hi - m2;
          }
          is += 2; u1 <<= 2; u2 <<= 2;
        }
      }
      return true;
    }
    case GgufFile::kQ6_K: {
      const std::int64_t nsb = n / 256;
      const std::uint8_t* p = src;
      for (std::int64_t sb = 0; sb < nsb; ++sb, p += 210) {
        const std::uint8_t* ql = p;
        const std::uint8_t* qh = p + 128;
        const auto* sc = reinterpret_cast<const std::int8_t*>(p + 192);
        std::uint16_t d16; std::memcpy(&d16, p + 208, 2);
        const float d = f16_to_f32_(d16);
        float* y = out + sb * 256;
        for (int half = 0; half < 2; ++half) {
          const int qlo = half * 64, qho = half * 32, sco = half * 8;
          const int yo = half * 128;
          for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int hi = qh[qho + l];
            const int q1 = ((ql[qlo + l]      & 0x0F) | (((hi >> 0) & 3) << 4))
                           - 32;
            const int q2 = ((ql[qlo + l + 32] & 0x0F) | (((hi >> 2) & 3) << 4))
                           - 32;
            const int q3 = ((ql[qlo + l]      >> 4 ) | (((hi >> 4) & 3) << 4))
                           - 32;
            const int q4 = ((ql[qlo + l + 32] >> 4 ) | (((hi >> 6) & 3) << 4))
                           - 32;
            y[yo + l]      = d * sc[sco + is + 0] * q1;
            y[yo + l + 32] = d * sc[sco + is + 2] * q2;
            y[yo + l + 64] = d * sc[sco + is + 4] * q3;
            y[yo + l + 96] = d * sc[sco + is + 6] * q4;
          }
        }
      }
      return true;
    }
    default:
      return false;
  }
}

std::int64_t
row_stride_bytes_(std::uint32_t type, std::int64_t in)
{
  BlkGeom bg = blk_geom_(type);
  if (bg.elems == 0) { return 0; }
  return (in / bg.elems) * bg.bytes;
}
}  // namespace

bool
GgufFile::dequant_row_f32(const Tensor& t, std::int64_t row,
                          float* out) const
{
  if (t.dims.size() != 2 || t.data == nullptr) { return false; }
  const std::int64_t in = t.dims[0];
  const std::int64_t outn = t.dims[1];
  if (row < 0 || row >= outn) { return false; }
  const std::int64_t stride = row_stride_bytes_(t.type, in);
  if (stride == 0) { return false; }
  return dequant_span_(t.type, t.data + row * stride, in, out);
}

bool
GgufFile::dequant_all_f32(const Tensor& t, float* out) const
{
  if (t.data == nullptr) { return false; }
  return dequant_span_(t.type, t.data, t.numel(), out);
}

}  // namespace vpipe::genai
