#include "apple-silicon/coreml/ane-emitter.h"

#include "apple-silicon/coreml/coreml-model-manager.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

namespace vpipe {

namespace {

namespace fs = std::filesystem;

constexpr std::uint32_t kEntryMagic  = 0xdeadbeefu;
constexpr std::size_t   kFileHeader  = 64;
constexpr std::size_t   kEntryHeader = 64;

// Entry headers sit on 64-byte boundaries; see blob_offsets().
constexpr std::size_t
align64_(std::size_t v)
{
  return (v + 63) & ~std::size_t{63};
}

// ---- a minimal protobuf writer --------------------------------------
//
// Only what a ModelDescription needs: varints, length-delimited fields
// and nesting. Emitting these by hand is smaller than taking a protobuf
// dependency for 200 bytes.

void
put_varint_(std::vector<unsigned char>& b, std::uint64_t v)
{
  while (v >= 0x80) {
    b.push_back((unsigned char)(v | 0x80));
    v >>= 7;
  }
  b.push_back((unsigned char)v);
}

void
put_tag_(std::vector<unsigned char>& b, int field, int wire)
{
  put_varint_(b, ((std::uint64_t)field << 3) | (std::uint64_t)wire);
}

void
put_bytes_(std::vector<unsigned char>& b, int field,
           const std::vector<unsigned char>& v)
{
  put_tag_(b, field, 2);
  put_varint_(b, v.size());
  b.insert(b.end(), v.begin(), v.end());
}

void
put_str_(std::vector<unsigned char>& b, int field, const std::string& s)
{
  put_tag_(b, field, 2);
  put_varint_(b, s.size());
  b.insert(b.end(), s.begin(), s.end());
}

// CoreML ArrayFeatureType.ArrayDataType. FLOAT16 is 65552, which is why
// it encodes as the three-byte varint 90 80 04 in an emitted model.
constexpr std::uint64_t kArrayFloat16 = 65552;

// FeatureDescription{ name, type: FeatureType{ multiArrayType } }.
std::vector<unsigned char>
feature_(const std::string& name, const std::vector<std::int64_t>& dims)
{
  std::vector<unsigned char> arr;      // ArrayFeatureType
  {
    std::vector<unsigned char> shape;  // packed repeated int64
    for (std::int64_t d : dims) { put_varint_(shape, (std::uint64_t)d); }
    put_bytes_(arr, 1, shape);
    put_tag_(arr, 2, 0);
    put_varint_(arr, kArrayFloat16);
  }
  std::vector<unsigned char> type;      // FeatureType
  put_bytes_(type, 5, arr);             // multiArrayType
  std::vector<unsigned char> fd;
  put_str_(fd, 1, name);
  put_bytes_(fd, 3, type);
  return fd;
}

// The 83-byte preamble coremltools writes, with the protobuf's length
// at +0x4b. The two constants (0x1f6, 9) and the "generic" string are
// carried verbatim because their meaning is not documented; only the
// length varies.
void
put_header_(std::vector<unsigned char>& out, std::size_t proto_len)
{
  out.assign(0x53, 0);
  auto u32 = [&](std::size_t off, std::uint32_t v) {
    std::memcpy(out.data() + off, &v, sizeof(v));
  };
  auto u64 = [&](std::size_t off, std::uint64_t v) {
    std::memcpy(out.data() + off, &v, sizeof(v));
  };
  u32(0x00, 0x1f6);
  u32(0x04, 9);
  u32(0x1c, 7);                       // strlen("generic")
  std::memcpy(out.data() + 0x24, "generic", 7);
  u32(0x2b, 9);
  u64(0x4b, (std::uint64_t)proto_len);
}

}  // namespace

bool
AneGraphSpec::valid() const
{
  if (M <= 0 || K <= 0 || N <= 0) { return false; }
  if (bk < 0 || bn < 0) { return false; }
  // Runtime weights take a narrower LAST tile where a width does not
  // divide: every tile is sliced from the whole input at its own bounds.
  // Constant tiles stay whole, which is what template blobs assume.
  if (!runtime_weights) {
    if (bk > 0 && K % bk != 0) { return false; }
    if (bn > 0 && N % bn != 0) { return false; }
    if (kind != Kind::Matmul && bn > 0 && K % bn != 0) {
      // the down projection's N is K, and it takes the same bn
      return false;
    }
  }
  // [out,in] is a property of an INPUT; a constant's layout belongs to
  // CoreML, and its blob carries [in,out] tiles.
  if (weights_out_in && !runtime_weights) { return false; }
  if (biases && (!runtime_weights || kind == Kind::Matmul)) {
    return false;
  }
  return true;
}

std::string
AneEmitter::output_name(const AneGraphSpec&)
{
  // FIXED, because emit_mil_() names its final op this and nothing
  // else. It used to be DERIVED from the tiling, which was wrong the
  // moment a shape tiled: the intermediate names carry a running
  // counter, so no rule outside the emitter can predict which one lands
  // last. A caller then bound an output that did not exist -- the model
  // still loaded and still ran, and returned nothing useful.
  return "out";
}

std::vector<std::size_t>
AneEmitter::weight_slabs(const AneGraphSpec& s)
{
  std::vector<std::size_t> out;
  if (!s.valid()) { return out; }
  if (s.runtime_weights) { return out; }   // the matrices are inputs
  auto push_tiles = [&](int K, int N) {
    const int bk = (s.bk > 0) ? std::min(s.bk, K) : K;
    const int bn = (s.bn > 0) ? std::min(s.bn, N) : N;
    for (int j = 0; j < (N + bn - 1) / bn; ++j) {
      const int n = std::min(bn, N - j * bn);
      for (int t = 0; t < (K + bk - 1) / bk; ++t) {
        const int k = std::min(bk, K - t * bk);
        out.push_back((std::size_t)k * (std::size_t)n * 2);
      }
    }
  };
  if (s.kind == AneGraphSpec::Kind::Matmul) {
    push_tiles(s.K, s.N);
  } else if (s.kind == AneGraphSpec::Kind::GeluFfn) {
    push_tiles(s.K, s.N);   // up
    push_tiles(s.N, s.K);   // down
  } else {
    push_tiles(s.K, s.N);   // gate
    push_tiles(s.K, s.N);   // up
    push_tiles(s.N, s.K);   // down
  }
  return out;
}

std::vector<AneEmitter::Feature>
AneEmitter::input_features(const AneGraphSpec& s)
{
  std::vector<Feature> f;
  f.push_back({input_name(), {s.M, s.K}});
  if (!s.runtime_weights) { return f; }
  // [in,out] unless weights_out_in, which swaps every one.
  auto dims = [&](int in, int out) -> std::vector<std::int64_t> {
    return s.weights_out_in ? std::vector<std::int64_t>{out, in}
                            : std::vector<std::int64_t>{in, out};
  };
  if (s.kind == AneGraphSpec::Kind::Matmul) {
    f.push_back({"w", dims(s.K, s.N)});
  } else if (s.kind == AneGraphSpec::Kind::GeluFfn) {
    f.push_back({"wu", dims(s.K, s.N)});
    f.push_back({"wd", dims(s.N, s.K)});
    if (s.biases) {
      f.push_back({"bu", {1, s.N}});
      f.push_back({"bd", {1, s.K}});
    }
  } else {
    f.push_back({"wg", dims(s.K, s.N)});
    f.push_back({"wu", dims(s.K, s.N)});
    f.push_back({"wd", dims(s.N, s.K)});
    if (s.biases) {
      f.push_back({"bg", {1, s.N}});
      f.push_back({"bu", {1, s.N}});
      f.push_back({"bd", {1, s.K}});
    }
  }
  return f;
}

std::size_t
AneEmitter::input_elems(const AneGraphSpec& s)
{
  return (std::size_t)s.M * (std::size_t)s.K;
}

std::size_t
AneEmitter::output_elems(const AneGraphSpec& s)
{
  return (std::size_t)s.M *
         (std::size_t)(s.kind == AneGraphSpec::Kind::Matmul ? s.N : s.K);
}

std::string
AneEmitter::emit_mil_(const AneGraphSpec& s)
{
  std::ostringstream o;
  o << mil_preamble()
    << "    func main<ios18>(tensor<fp16, [" << s.M << ", " << s.K
    << "]> x";
  // Runtime weights are further function inputs, declared whole; the
  // tiles below slice them.
  {
    const std::vector<Feature> f = input_features(s);
    for (std::size_t i = 1; i < f.size(); ++i) {
      o << ", tensor<fp16, [" << f[i].shape[0] << ", " << f[i].shape[1]
        << "]> " << f[i].name;
    }
  }
  o << ") {\n";
  // Which whole input a projection's tiles are cut from: the tiled lambda
  // is called with the projection prefix, and a plain matmul has none.
  auto whole_of = [](const std::string& pfx) -> std::string {
    return pfx.empty() ? std::string("w") : "w" + pfx;
  };

  std::size_t blob_off = kFileHeader;   // next ENTRY HEADER position
  int uid = 0;

  // One tiled matmul: `src` [M,K] @ W[K,N] -> a variable name.
  auto tiled = [&](const std::string& src, int K, int N,
                   const std::string& pfx, const std::string& final_name) {
    const int bk = (s.bk > 0) ? std::min(s.bk, K) : K;
    const int bn = (s.bn > 0) ? std::min(s.bn, N) : N;
    const int nk = (K + bk - 1) / bk;
    const int nn = (N + bn - 1) / bn;
    // When there is one column the final value is that column's last
    // op, so it must carry `final_name` directly -- there is no concat
    // to rename it.
    const bool one_col = (nn == 1);
    std::vector<std::string> cols;
    for (int j = 0; j < nn; ++j) {
      const int n0 = j * bn, n = std::min(bn, N - n0);
      std::string acc;
      for (int t = 0; t < nk; ++t) {
        const int k0 = t * bk, k = std::min(bk, K - k0);
        std::string xin = src;
        if (nk > 1) {
          const std::string sl = pfx + "_sl" + std::to_string(uid++);
          o << "            tensor<int32, [2]> " << sl
            << "_b = const()[name = string(\"" << sl
            << "_b\"), val = tensor<int32, [2]>([0, " << k0 << "])];\n"
            << "            tensor<int32, [2]> " << sl
            << "_e = const()[name = string(\"" << sl
            << "_e\"), val = tensor<int32, [2]>([" << s.M << ", "
            << (k0 + k) << "])];\n"
            << "            tensor<fp16, [" << s.M << ", " << k << "]> "
            << sl << " = slice_by_index(begin = " << sl << "_b, end = "
            << sl << "_e, x = " << src << ")[name = string(\"" << sl
            << "\")];\n";
          xin = sl;
        }
        const bool last_op = one_col && (t == nk - 1);
        const std::string mm =
            (last_op && nk == 1) ? final_name
                                 : (pfx + "mm_" + std::to_string(uid++));
        if (s.runtime_weights) {
          // Cut the tile from the whole input: rows k0..k0+k, columns
          // n0..n0+n of an [in,out] one -- the tile a constant would carry
          // -- or that tile transposed, out of an [out,in] one, which the
          // matmul then reads with transpose_y.
          const bool oi = s.weights_out_in;
          const int r0 = oi ? n0 : k0, c0 = oi ? k0 : n0;
          const int tr = oi ? n : k, tc = oi ? k : n;
          o << "            tensor<int32, [2]> " << mm
            << "_wb = const()[name = string(\"" << mm
            << "_wb\"), val = tensor<int32, [2]>([" << r0 << ", " << c0
            << "])];\n"
            << "            tensor<int32, [2]> " << mm
            << "_we = const()[name = string(\"" << mm
            << "_we\"), val = tensor<int32, [2]>([" << (r0 + tr) << ", "
            << (c0 + tc) << "])];\n"
            << "            tensor<fp16, [" << tr << ", " << tc << "]> " << mm
            << "_w = slice_by_index(begin = " << mm << "_wb, end = " << mm
            << "_we, x = " << whole_of(pfx) << ")[name = string(\"" << mm
            << "_w\")];\n";
        } else {
          o << "            tensor<fp16, [" << k << ", " << n << "]> " << mm
            << "_w = const()[name = string(\"" << mm
            << "_w\"), val = tensor<fp16, [" << k << ", " << n
            << "]>(BLOBFILE(path = string(\"@model_path/weights/"
               "weight.bin\"), offset = uint64(" << blob_off << ")))];\n";
        }
        o << "            bool " << mm
          << "_tx = const()[name = string(\"" << mm
          << "_tx\"), val = bool(false)];\n"
          << "            bool " << mm
          << "_ty = const()[name = string(\"" << mm
          << "_ty\"), val = bool("
          << (s.weights_out_in ? "true" : "false") << ")];\n"
          << "            tensor<fp16, [" << s.M << ", " << n << "]> " << mm
          << " = matmul(transpose_x = " << mm << "_tx, transpose_y = " << mm
          << "_ty, x = " << xin << ", y = " << mm
          << "_w)[name = string(\"" << mm << "\")];\n";
        blob_off = align64_(blob_off + kEntryHeader +
                            (std::size_t)k * (std::size_t)n * 2);
        if (acc.empty()) {
          acc = mm;
        } else {
          const std::string ad =
              (one_col && t == nk - 1) ? final_name
                                       : (pfx + "acc_" + std::to_string(uid++));
          o << "            tensor<fp16, [" << s.M << ", " << n << "]> "
            << ad << " = add(x = " << acc << ", y = " << mm
            << ")[name = string(\"" << ad << "\")];\n";
          acc = ad;
        }
      }
      cols.push_back(acc);
    }
    if (cols.size() == 1u) { return cols[0]; }
    const std::string cc = final_name;
    o << "            int32 " << cc
      << "_ax = const()[name = string(\"" << cc
      << "_ax\"), val = int32(1)];\n"
      << "            bool " << cc
      << "_il = const()[name = string(\"" << cc
      << "_il\"), val = bool(false)];\n"
      << "            tensor<fp16, [" << s.M << ", " << N << "]> " << cc
      << " = concat(axis = " << cc << "_ax, interleave = " << cc
      << "_il, values = (";
    for (std::size_t i = 0; i < cols.size(); ++i) {
      o << (i ? ", " : "") << cols[i];
    }
    o << "))[name = string(\"" << cc << "\")];\n";
    return cc;
  };

  std::string last;
  if (s.kind == AneGraphSpec::Kind::Matmul) {
    last = tiled("x", s.K, s.N, "", "out");
  } else if (s.kind == AneGraphSpec::Kind::GeluFfn) {
    std::string u = tiled("x", s.K, s.N, "u", "uout");
    if (s.biases) {
      o << "            tensor<fp16, [" << s.M << ", " << s.N
        << "]> ub = add(x = " << u << ", y = bu)[name = string(\"ub\")];\n";
      u = "ub";
    }
    // GELU SPELLED OUT: torch's approximate="tanh",
    // 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), as elementwise ops.
    //
    // For the reason SiLU is (see the SwiGLU branch below): the compiler's
    // `gelu` op goes to a low-precision form once the graph is compiled
    // for 2048 rows, the tiers' chunk. MEASURED on an M4 Pro, the op in
    // isolation: max abs error 6.0e-3 over |x| <= 8 (8% relative on small
    // values) against 1.8e-3 spelled out; at LTX-2.5's feed-forward
    // (4096 x 16384, 6144 ANE rows) the whole FF's error against fp64
    // halves, 7.1e-3 -> 3.3e-3, for +9.3% ANE time (27.0 -> 29.5 ms per
    // 1k rows). Nothing inside overflows: x^3 saturating to inf takes
    // tanh to +-1, never inf - inf. VPIPE_ANE_GELU_OP=1 emits the op
    // instead, for A/B.
    static const bool kGeluOp = std::getenv("VPIPE_ANE_GELU_OP") != nullptr;
    if (!kGeluOp) {
      const std::string T =
          "tensor<fp16, [" + std::to_string(s.M) + ", " +
          std::to_string(s.N) + "]> ";
      auto cst = [&](const char* nm, double v) {
        o << "            fp16 " << nm << " = const()[name = string(\"" << nm
          << "\"), val = fp16(" << v << ")];\n";
      };
      cst("g_c3", 0.044715);
      cst("g_kc", std::sqrt(2.0 / M_PI));
      cst("g_one", 1.0);
      cst("g_half", 0.5);
      auto op = [&](const char* nm, const std::string& expr) {
        o << "            " << T << nm << " = " << expr
          << "[name = string(\"" << nm << "\")];\n";
      };
      op("g_x2", "mul(x = " + u + ", y = " + u + ")");
      op("g_x3", "mul(x = g_x2, y = " + u + ")");
      op("g_x3c", "mul(x = g_x3, y = g_c3)");
      op("g_in", "add(x = " + u + ", y = g_x3c)");
      op("g_ink", "mul(x = g_in, y = g_kc)");
      op("g_th", "tanh(x = g_ink)");
      op("g_1p", "add(x = g_th, y = g_one)");
      op("g_hx", "mul(x = " + u + ", y = g_half)");
      op("hh", "mul(x = g_hx, y = g_1p)");
    } else {
      // The tanh approximation, as the checkpoint's reference computes it.
      o << "            string gm = const()[name = string(\"gm\"), val = "
           "string(\"TANH_APPROXIMATION\")];\n"
        << "            tensor<fp16, [" << s.M << ", " << s.N
        << "]> hh = gelu(mode = gm, x = " << u
        << ")[name = string(\"hh\")];\n";
    }
    if (s.biases) {
      const std::string d = tiled("hh", s.N, s.K, "d", "dout");
      o << "            tensor<fp16, [" << s.M << ", " << s.K
        << "]> out = add(x = " << d << ", y = bd)[name = string(\"out\")];\n";
      last = "out";
    } else {
      last = tiled("hh", s.N, s.K, "d", "out");
    }
  } else {
    std::string g = tiled("x", s.K, s.N, "g", "gout");
    std::string u = tiled("x", s.K, s.N, "u", "uout");
    // A bias is one broadcast add after its projection's tiles are summed.
    auto biased = [&](const std::string& v, const std::string& b, int n,
                      const std::string& nm) {
      o << "            tensor<fp16, [" << s.M << ", " << n << "]> " << nm
        << " = add(x = " << v << ", y = " << b << ")[name = string(\""
        << nm << "\")];\n";
      return nm;
    };
    if (s.biases) {
      g = biased(g, "bg", s.N, "gb");
      u = biased(u, "bu", s.N, "ub");
    }
    // SiLU AS THE TWO-PATH STABLE SIGMOID, spelled out, branch-free:
    //   silu(g) = g * exp(min(g, 0)) / (1 + exp(-|g|))
    // -- exp(-g) above zero and exp(g) below, so neither exponent is ever
    // positive and neither exp can overflow.
    //
    // Spelled out because the compiler's own `silu` (and `sigmoid`, which
    // it lowers the same way) goes to a low-precision form once the graph
    // is compiled for 1024 or more rows -- which every tier's 2048-row
    // chunk is. MEASURED on an M4 Pro, one op in isolation over fp16
    // inputs: 11% relative error on small values (max abs 1.5e-2 over
    // |x| <= 8) against 0.1% (4.7e-3) for this form, which matches what
    // the op gives at 512 rows. On a Krea-2 1024^2 image with the ANE
    // feed-forward, 35.2 -> 39.3 dB against GPU-only; the module is 5-8%
    // slower and a whole step ~0.5%, the extra ANE time hiding under the
    // GPU's rows. VPIPE_ANE_SILU_OP=1 emits the op instead, for A/B.
    static const bool kSiluOp = std::getenv("VPIPE_ANE_SILU_OP") != nullptr;
    const std::string T = "tensor<fp16, [" + std::to_string(s.M) + ", " +
                          std::to_string(s.N) + "]> ";
    auto cst = [&](const char* nm, double v) {
      o << "            fp16 " << nm << " = const()[name = string(\"" << nm
        << "\"), val = fp16(" << v << ")];\n";
    };
    auto op = [&](const char* nm, const std::string& expr) {
      o << "            " << T << nm << " = " << expr << "[name = string(\""
        << nm << "\")];\n";
    };
    if (kSiluOp) {
      op("sg", "silu(x = " + g + ")");
    } else {
      cst("s_zero", 0.0);
      cst("s_one", 1.0);
      cst("s_neg", -1.0);
      op("s_min", "minimum(x = " + g + ", y = s_zero)");
      op("s_num", "exp(x = s_min)");
      op("s_abs", "abs(x = " + g + ")");
      op("s_nab", "mul(x = s_abs, y = s_neg)");
      op("s_e", "exp(x = s_nab)");
      op("s_den", "add(x = s_e, y = s_one)");
      op("sgm", "real_div(x = s_num, y = s_den)");
      op("sg", "mul(x = " + g + ", y = sgm)");
    }
    op("hh", "mul(x = sg, y = " + u + ")");
    if (s.biases) {
      last = biased(tiled("hh", s.N, s.K, "d", "dout"), "bd", s.K, "out");
    } else {
      last = tiled("hh", s.N, s.K, "d", "out");
    }
  }
  o << "        } -> (" << last << ");\n}";
  return o.str();
}

std::vector<unsigned char>
AneEmitter::emit_description_(std::span<const Feature> inputs,
                              std::span<const Feature> outputs)
{
  std::vector<unsigned char> desc;
  for (const Feature& f : inputs) {
    put_bytes_(desc, 1, feature_(f.name, f.shape));     // input
  }
  for (const Feature& f : outputs) {
    put_bytes_(desc, 10, feature_(f.name, f.shape));    // output
  }

  std::vector<unsigned char> out;
  put_header_(out, desc.size());
  out.insert(out.end(), desc.begin(), desc.end());
  // The 16-byte trailer: the same tag the file opens with, then zeros.
  std::vector<unsigned char> tail(16, 0);
  const std::uint32_t tag = 0x1f6;
  std::memcpy(tail.data(), &tag, sizeof(tag));
  out.insert(out.end(), tail.begin(), tail.end());
  return out;
}

std::vector<unsigned char>
AneEmitter::emit_blob_(std::span<const AneBlobSlab> weights)
{
  std::vector<std::size_t> sizes;
  sizes.reserve(weights.size());
  for (const auto& w : weights) { sizes.push_back(w.bytes); }
  const std::vector<std::size_t> offs = blob_offsets(sizes);
  // Not padded after the last slab, as coremltools writes it.
  const std::size_t total =
      offs.empty() ? kFileHeader
                   : offs.back() + kEntryHeader + sizes.back();
  std::vector<unsigned char> b(total, 0);
  auto u32 = [&](std::size_t off, std::uint32_t v) {
    std::memcpy(b.data() + off, &v, sizeof(v));
  };
  auto u64 = [&](std::size_t off, std::uint64_t v) {
    std::memcpy(b.data() + off, &v, sizeof(v));
  };
  u32(0, (std::uint32_t)weights.size());
  u32(4, 2);
  for (std::size_t i = 0; i < weights.size(); ++i) {
    const std::size_t off = offs[i];
    const AneBlobSlab& w  = weights[i];
    u32(off, kEntryMagic);
    u32(off + 4, 1);
    u64(off + 8, (std::uint64_t)w.bytes);
    u64(off + 16, (std::uint64_t)(off + kEntryHeader));
    if (w.bytes > 0 && w.data != nullptr) {
      std::memcpy(b.data() + off + kEntryHeader, w.data, w.bytes);
    }
  }
  return b;
}

std::vector<std::size_t>
AneEmitter::blob_offsets(std::span<const std::size_t> bytes)
{
  std::vector<std::size_t> out;
  out.reserve(bytes.size());
  std::size_t off = kFileHeader;
  for (std::size_t n : bytes) {
    out.push_back(off);
    off = align64_(off + kEntryHeader + n);
  }
  return out;
}

const char*
AneEmitter::mil_preamble() noexcept
{
  return "program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", "
         "\"3520.4.1\"}, {\"coremlc-version\", \"3520.5.1\"}})]\n"
         "{\n";
}

bool
AneEmitter::emit(const AneGraphSpec& s, std::span<const AneBlobSlab> weights,
                 const std::string& dir, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (!s.valid()) { return fail("invalid graph spec"); }
  const std::vector<std::size_t> want = weight_slabs(s);
  if (want.size() != weights.size()) {
    return fail("graph wants " + std::to_string(want.size()) +
                " weight slabs, caller gave " +
                std::to_string(weights.size()));
  }
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (want[i] != weights[i].bytes) {
      return fail("slab " + std::to_string(i) + " is " +
                  std::to_string(weights[i].bytes) + " bytes, graph wants " +
                  std::to_string(want[i]));
    }
  }

  const int out_cols =
      (s.kind == AneGraphSpec::Kind::Matmul) ? s.N : s.K;
  const std::vector<Feature> in = input_features(s);
  const Feature out[1] = {{output_name(s), {s.M, out_cols}}};
  return write_bundle(dir, emit_mil_(s), in, out, weights, err);
}

bool
AneEmitter::write_bundle(const std::string& dir, const std::string& mil,
                         std::span<const Feature> inputs,
                         std::span<const Feature> outputs,
                         std::span<const AneBlobSlab> weights,
                         std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (inputs.empty() || outputs.empty()) {
    return fail("a model needs at least one input and one output");
  }
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(weights.empty() ? fs::path(dir)
                                         : fs::path(dir) / "weights",
                         ec);
  if (ec) { return fail("cannot create " + dir + ": " + ec.message()); }
  {
    std::ofstream f(fs::path(dir) / "model.mil", std::ios::binary);
    f.write(mil.data(), (std::streamsize)mil.size());
    if (!f) { return fail("cannot write model.mil"); }
  }
  {
    const std::vector<unsigned char> d = emit_description_(inputs, outputs);
    std::ofstream f(fs::path(dir) / "coremldata.bin", std::ios::binary);
    f.write((const char*)d.data(), (std::streamsize)d.size());
    if (!f) { return fail("cannot write coremldata.bin"); }
  }
  if (!weights.empty()) {
    const std::vector<unsigned char> b = emit_blob_(weights);
    std::ofstream f(fs::path(dir) / "weights" / "weight.bin",
                    std::ios::binary);
    f.write((const char*)b.data(), (std::streamsize)b.size());
    if (!f) { return fail("cannot write weight.bin"); }
  }
  return true;
}

bool
AneEmitter::self_test(CoreMLModelManager& mgr, const std::string& scratch,
                      std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  // Small on purpose: this runs before the tier is trusted, and what is
  // under test is the FORMAT, which CoreML parses identically whichever
  // engine it then picks. A 64-cube matmul settles that in milliseconds.
  AneGraphSpec s;
  s.kind = AneGraphSpec::Kind::Matmul;
  s.M = s.K = s.N = 64;

  std::vector<std::uint16_t> w((std::size_t)s.K * s.N);
  std::vector<std::uint16_t> x((std::size_t)s.M * s.K);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> d(-0.5f, 0.5f);
  std::vector<float> wf(w.size()), xf(x.size());
  for (std::size_t i = 0; i < w.size(); ++i) {
    wf[i] = d(rng) * 0.25f;
    const _Float16 h = (_Float16)wf[i];
    std::memcpy(&w[i], &h, sizeof(h));
    wf[i] = (float)h;
  }
  for (std::size_t i = 0; i < x.size(); ++i) {
    xf[i] = d(rng) * 0.5f;
    const _Float16 h = (_Float16)xf[i];
    std::memcpy(&x[i], &h, sizeof(h));
    xf[i] = (float)h;
  }
  const AneBlobSlab slab{w.data(), w.size() * 2};
  const std::string dir = scratch + "/ane-emitter-selftest.mlmodelc";
  if (!emit(s, std::span<const AneBlobSlab>(&slab, 1), dir, err)) {
    return false;
  }

  auto model = mgr.load(dir, 3);
  std::error_code ec;
  if (model == nullptr || !model->valid()) {
    fs::remove_all(dir, ec);
    return fail("emitted model did not load");
  }

  std::vector<std::uint16_t> y((std::size_t)s.M * s.N, 0);
  CoreMLPredictInput in;
  in.name  = input_name();
  in.data  = x.data();
  in.dtype = CoreMLDType::F16;
  in.shape = {s.M, s.K};
  CoreMLPredictOutput out;
  out.name          = output_name(s);
  out.want          = CoreMLDType::F16;
  out.backing       = y.data();
  out.backing_elems = y.size();
  const CoreMLPredictInput ins[1]  = {in};
  CoreMLPredictOutput      outs[1] = {out};
  const bool ok = model->predict(ins, outs);
  model.reset();
  fs::remove_all(dir, ec);
  if (!ok) { return fail("emitted model did not predict"); }

  // NUMBERS, not just a successful load. A format that has drifted can
  // still parse and still run while reading the weights from the wrong
  // offset -- which is exactly the failure that must not reach a user.
  double num = 0.0, den = 0.0;
  for (int m = 0; m < s.M; ++m) {
    for (int n = 0; n < s.N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < s.K; ++k) {
        acc += (double)xf[(std::size_t)m * s.K + k] *
               (double)wf[(std::size_t)k * s.N + n];
      }
      _Float16 h;
      std::memcpy(&h, &y[(std::size_t)m * s.N + n], sizeof(h));
      const double got = (double)h;
      num += (got - acc) * (got - acc);
      den += acc * acc;
    }
  }
  const double rel = (den > 0.0) ? std::sqrt(num / den) : 1.0;
  // fp16 accumulation over K=64 lands near 1e-3; a wrong offset or a
  // permuted layout lands at O(1), so the bar separates them by orders
  // of magnitude rather than finely.
  if (!(rel < 5e-2)) {
    char buf[128];
    std::snprintf(buf, sizeof(buf),
                  "emitted model computed the wrong result (rel-L2 %.3e)",
                  rel);
    return fail(buf);
  }
  return true;
}

}  // namespace vpipe
