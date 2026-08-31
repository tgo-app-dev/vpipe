#include "generative-models/shared/runtime-lora.h"

#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "interfaces/session-context-intf.h"

#include <cstring>
#include <vector>
#include <algorithm>

namespace vpipe {
namespace genai {
namespace lora {

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

std::uint16_t
f32_to_bf16(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

float
bf16_to_f32(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// A 1-element tensor as float, whatever it is stored as. Used for the
// per-module `alpha`.
bool
scalar_f32(const MetalLlamaWeights& w, const std::string& name,
           const MetalLlamaWeights::TensorInfo* ti, MetalCompute* mc,
           float* out)
{
  std::size_t n = 1;
  for (auto d : ti->shape) { n *= (std::size_t)d; }
  if (n != 1) { return false; }
  SharedBuffer b = w.load(name, mc);
  if (b.empty()) { return false; }
  if (ti->dtype == "F32") {
    *out = *static_cast<const float*>(b.contents());
    return true;
  }
  if (ti->dtype == "F16") {
    *out = (float)*static_cast<const _Float16*>(b.contents());
    return true;
  }
  if (ti->dtype == "BF16") {
    *out = bf16_to_f32(*static_cast<const std::uint16_t*>(b.contents()));
    return true;
  }
  return false;
}

// Publishers disagree about one thing only: whether the module names
// carry a container prefix. ComfyUI-convention adapters key on
// `diffusion_model.<module>` where a model's own names have no prefix at
// all, and the shapes and the ORDER are otherwise identical -- so this
// is a lookup with two spellings rather than a conversion. An adapter
// that needs more than a prefix is not one of these and must not be
// coerced into looking like one.
const char* const kPrefixes[] = {"", "diffusion_model."};

// Two spellings of a factor tensor are in the wild and they differ by
// an infix: `<module>.lora_A.weight`, and peft's
// `<module>.lora_A.<adapter>.weight` where the adapter is usually
// "default". Rather than try both at every lookup -- which would only
// ever cover the adapter names guessed at -- a file's spelling is
// DISCOVERED once from its own tensor names.
const char kMark[] = ".lora_A.";

// The module a factor tensor belongs to, plus the suffix that followed
// it. Empty module when the name is not an A factor. B is not scanned:
// a module without its B is unbindable anyway, and bind() checks both.
std::string
split_factor(const std::string& tensor, std::string* suffix)
{
  const std::size_t p = tensor.find(kMark);
  if (p == std::string::npos) { return {}; }
  if (suffix != nullptr) { *suffix = tensor.substr(p); }
  return tensor.substr(0, p);
}

// The module part of an A-factor tensor, given this file's suffix.
std::string
module_of(const std::string& tensor, const std::string& suf)
{
  if (tensor.size() <= suf.size() ||
      tensor.compare(tensor.size() - suf.size(), suf.size(), suf) != 0) {
    return {};
  }
  return tensor.substr(0, tensor.size() - suf.size());
}

// ".lora_A.<x>.weight" -> ".lora_B.<x>.weight".
std::string
b_of_a(std::string suf)
{
  const std::size_t p = suf.find("lora_A");
  if (p != std::string::npos) { suf[p + 5] = 'B'; }
  return suf;
}

}  // namespace

std::unique_ptr<Adapter>
Adapter::open(const std::string& path, MetalCompute* mc, std::string* err,
              Rename rename)
{
  auto w = MetalLlamaWeights::open(path);
  if (!w.has_value()) {
    if (err != nullptr) { *err = "cannot open " + path; }
    return nullptr;
  }
  std::unique_ptr<Adapter> a(new Adapter());
  a->_mc = mc;
  a->_w = std::make_unique<MetalLlamaWeights>(std::move(*w));
  a->_rename = rename;
  a->index_suffix_();
  if (rename != nullptr) { a->index_renames_(); }
  return a;
}

// Index the file's modules under the MODEL's spelling.
//
// Built once here rather than mapped per bind, because the map runs the
// wrong way for that: it turns a FILE name into a model name, and bind
// arrives holding the model name. Inverting it per call would mean
// walking every tensor name per module -- and inverting it by hand
// would mean a second map to keep right, which is the thing
// shared/lora-names.h exists to avoid.
//
// A name the rename does not recognise is skipped rather than recorded
// empty, and a name it maps to ITSELF adds nothing: the direct lookup
// already finds those, and an entry would only make a hit ambiguous.
// This file's factor spelling, and its file-level alpha.
//
// The alpha is read here because it is the ONE number a diffusers
// export states outside the tensors: peft writes `alpha` into
// `__metadata__` and no per-module `.alpha` at all. An adapter read
// without it binds every module, reports success and applies at
// alpha == rank -- 16x the trained strength on the lightx2v files.
void
Adapter::index_suffix_()
{
  for (const std::string& t : _w->tensor_names()) {
    std::string suf;
    if (split_factor(t, &suf).empty()) { continue; }
    _suf_a = suf;
    _suf_b = b_of_a(suf);
    break;
  }
  const auto& md = _w->metadata();
  const auto it = md.find("alpha");
  if (it != md.end()) {
    try {
      _meta_alpha = std::stof(it->second);
    } catch (...) {
      _meta_alpha = 0.0f;
    }
  }
}

std::string
Adapter::factor_name_(const std::string& key, char ab) const
{
  return key + (ab == 'A' ? _suf_a : _suf_b);
}

void
Adapter::index_renames_()
{
  for (const std::string& t : _w->tensor_names()) {
    const std::string m = module_of(t, _suf_a);
    if (m.empty()) { continue; }
    const std::string mapped = _rename(m);
    if (mapped.empty() || mapped == m) { continue; }
    _by_renamed.emplace(mapped, m);
  }
}

Adapter::~Adapter() = default;

// Resolve a MODEL module name to the key the FILE uses, or "" when the
// file carries neither spelling of it.
std::string
Adapter::resolve_(const std::string& module, bool* via_rename) const
{
  if (via_rename != nullptr) { *via_rename = false; }
  auto both = [&](const std::string& kk) {
    return _w->info(factor_name_(kk, 'A')) != nullptr &&
           _w->info(factor_name_(kk, 'B')) != nullptr;
  };
  for (const char* pre : kPrefixes) {
    const std::string kk = std::string(pre) + module;
    if (both(kk)) { return kk; }
  }
  // The file's OWN spelling wins; the rename is the fallback. A file
  // carrying both conventions for one module is not a thing that
  // happens -- they are different publishers' names for one adapter --
  // but if it did, the name the model uses is the less surprising
  // answer.
  const auto it = _by_renamed.find(module);
  if (it != _by_renamed.end() && both(it->second)) {
    if (via_rename != nullptr) { *via_rename = true; }
    return it->second;
  }
  return {};
}

bool
Adapter::has(const std::string& module) const
{
  return !resolve_(module, nullptr).empty();
}

// `key`'s rank, and the multiplier that folds this module's OWN
// strength into A.
//
// kohya / ai-toolkit ship a per-module `alpha` and the update is scaled
// by alpha/rank. A diffusers/peft export states one alpha for the whole
// FILE instead. Absent from both means the factors are already at
// strength -- the same thing as alpha == rank -- which is what a
// diffusers LoRA with no metadata means and NOT what one with metadata
// means, hence the fallback rather than a single default.
bool
Adapter::factor_meta_(const std::string& key, int* rank, float* mul)
{
  const auto* ai = _w->info(factor_name_(key, 'A'));
  const auto* bi = _w->info(factor_name_(key, 'B'));
  if (ai == nullptr || bi == nullptr) { return false; }
  if (ai->shape.size() != 2 || bi->shape.size() != 2 ||
      ai->shape[0] != bi->shape[1]) {
    return false;
  }
  *rank = (int)ai->shape[0];
  *mul = 1.0f;
  if (*rank <= 0) { return false; }
  if (const auto* al = _w->info(key + ".alpha")) {
    float v = 0.0f;
    if (scalar_f32(*_w, key + ".alpha", al, _mc, &v)) {
      *mul = v / (float)*rank;
    }
  } else if (_meta_alpha > 0.0f) {
    *mul = _meta_alpha / (float)*rank;
  }
  return true;
}

// One factor, converted to bf16 and (for A) pre-scaled -- folding the
// file's strength here keeps the kernels scale-free and costs one pass
// over the smaller of the two matrices.
SharedBuffer
Adapter::take_(const std::string& name, float m)
{
  const auto* ti = _w->info(name);
  if (ti == nullptr || ti->shape.size() != 2) { return {}; }
  const std::size_t cnt =
      (std::size_t)ti->shape[0] * (std::size_t)ti->shape[1];
  SharedBuffer src = _w->load(name, _mc);
  if (src.empty()) { return {}; }
  SharedBuffer dst = _mc->make_shared_buffer(cnt * 2);
  if (dst.empty()) { return {}; }
  auto* d = static_cast<std::uint16_t*>(dst.contents());
  if (ti->dtype == "BF16") {
    const auto* p = static_cast<const std::uint16_t*>(src.contents());
    for (std::size_t i = 0; i < cnt; ++i) {
      d[i] = m == 1.0f ? p[i] : f32_to_bf16(bf16_to_f32(p[i]) * m);
    }
  } else if (ti->dtype == "F16") {
    const auto* p = static_cast<const _Float16*>(src.contents());
    for (std::size_t i = 0; i < cnt; ++i) {
      d[i] = f32_to_bf16((float)p[i] * m);
    }
  } else if (ti->dtype == "F32") {
    const auto* p = static_cast<const float*>(src.contents());
    for (std::size_t i = 0; i < cnt; ++i) { d[i] = f32_to_bf16(p[i] * m); }
  } else {
    return {};
  }
  return dst;
}

bool
Adapter::bind(const std::string& module, int n, int k, Factors* out)
{
  if (out == nullptr || _w == nullptr) { return false; }
  bool via_rename = false;
  const std::string key = resolve_(module, &via_rename);
  if (key.empty()) { return false; }          // simply absent
  const auto* ai = _w->info(factor_name_(key, 'A'));
  const auto* bi = _w->info(factor_name_(key, 'B'));
  int rank = 0;
  float mul = 1.0f;
  if (!factor_meta_(key, &rank, &mul) ||
      ai->shape[1] != k || bi->shape[0] != n) {
    ++_skipped;
    return false;
  }
  out->rank = rank;
  out->a = take_(factor_name_(key, 'A'), mul);
  out->b = take_(factor_name_(key, 'B'), 1.0f);
  if (out->empty()) {
    *out = Factors{};
    ++_skipped;
    return false;
  }
  _max_rank = std::max(_max_rank, rank);
  ++_modules;
  if (via_rename) { ++_renamed; }
  return true;
}

bool
Adapter::bind_fused(const std::vector<std::string>& modules, int n, int k,
                    Factors* out, const RowMap& rows)
{
  if (out == nullptr || _w == nullptr || modules.empty() || !rows) {
    return false;
  }
  struct Part {
    std::string key;
    int rank = 0, n = 0;
    float mul = 1.0f;
  };
  std::vector<Part> parts;
  bool via_rename = false;
  int total_rank = 0, total_n = 0;
  for (const std::string& m : modules) {
    bool vr = false;
    Part p;
    p.key = resolve_(m, &vr);
    if (p.key.empty()) { return false; }       // absent: not a skip
    via_rename = via_rename || vr;
    const auto* ai = _w->info(factor_name_(p.key, 'A'));
    const auto* bi = _w->info(factor_name_(p.key, 'B'));
    if (!factor_meta_(p.key, &p.rank, &p.mul) || ai->shape[1] != k) {
      ++_skipped;
      return false;
    }
    p.n = (int)bi->shape[0];
    total_rank += p.rank;
    total_n += p.n;
    parts.push_back(std::move(p));
  }
  if (total_n != n) { ++_skipped; return false; }

  // A: the parts stacked on the RANK axis, each already carrying its
  // own alpha/rank. Rows are contiguous [rank, k] blocks, so this is a
  // concatenation and not a scatter.
  SharedBuffer a =
      _mc->make_shared_buffer((std::size_t)total_rank * (std::size_t)k * 2);
  SharedBuffer b =
      _mc->make_shared_buffer((std::size_t)n * (std::size_t)total_rank * 2);
  if (a.empty() || b.empty()) { return false; }
  auto* ad = static_cast<std::uint16_t*>(a.contents());
  auto* bd = static_cast<std::uint16_t*>(b.contents());
  // ZERO FIRST. Everything off the diagonal blocks stays zero, and that
  // is the whole content of "block diagonal" -- a buffer left as it was
  // allocated would add the previous tenant's bytes to two thirds of
  // the projection.
  std::memset(bd, 0, (std::size_t)n * (std::size_t)total_rank * 2);
  int r_off = 0;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    const Part& p = parts[i];
    SharedBuffer pa = take_(factor_name_(p.key, 'A'), p.mul);
    SharedBuffer pb = take_(factor_name_(p.key, 'B'), 1.0f);
    if (pa.empty() || pb.empty()) { ++_skipped; return false; }
    std::memcpy(ad + (std::size_t)r_off * (std::size_t)k, pa.contents(),
                (std::size_t)p.rank * (std::size_t)k * 2);
    const auto* src = static_cast<const std::uint16_t*>(pb.contents());
    for (int row = 0; row < p.n; ++row) {
      const int dst = rows((int)i, row);
      if (dst < 0 || dst >= n) { ++_skipped; return false; }
      std::memcpy(bd + (std::size_t)dst * (std::size_t)total_rank + r_off,
                  src + (std::size_t)row * (std::size_t)p.rank,
                  (std::size_t)p.rank * 2);
    }
    r_off += p.rank;
  }
  out->rank = total_rank;
  out->a = std::move(a);
  out->b = std::move(b);
  _max_rank = std::max(_max_rank, total_rank);
  ++_modules;
  if (via_rename) { ++_renamed; }
  return true;
}

bool
Adapter::block_diagonal_b(const Factors& f, int parts, int n)
{
  if (f.empty() || parts <= 1 || n <= 0) { return false; }
  if (f.rank % parts != 0 || n % parts != 0) { return false; }
  const int rb = f.rank / parts, nb = n / parts;
  if ((std::size_t)n * (std::size_t)f.rank * 2 > f.b.byte_size()) {
    return false;
  }
  const auto* b = static_cast<const std::uint16_t*>(f.b.contents());
  for (int row = 0; row < n; ++row) {
    const int band = row / nb;
    for (int c = 0; c < f.rank; ++c) {
      if (c / rb == band) { continue; }        // on the diagonal block
      if (b[(std::size_t)row * f.rank + c] != 0) { return false; }
    }
  }
  return true;
}

bool
Adapter::permute_b_rows(Factors* f, int n, const RowMap& rows)
{
  if (f == nullptr || f->empty() || n <= 0 || !rows) { return false; }
  const std::size_t cnt = (std::size_t)n * (std::size_t)f->rank;
  if (cnt * 2 > f->b.byte_size()) { return false; }
  std::vector<std::uint16_t> tmp(cnt, 0);
  const auto* src = static_cast<const std::uint16_t*>(f->b.contents());
  std::vector<bool> hit((std::size_t)n, false);
  for (int r = 0; r < n; ++r) {
    const int d = rows(0, r);
    // NOT a permutation -- leave the factors untouched rather than
    // write a partly-scrambled B, which would be worse than the order
    // that was already there.
    if (d < 0 || d >= n || hit[(std::size_t)d]) { return false; }
    hit[(std::size_t)d] = true;
    std::memcpy(tmp.data() + (std::size_t)d * f->rank,
                src + (std::size_t)r * f->rank,
                (std::size_t)f->rank * 2);
  }
  std::memcpy(f->b.contents(), tmp.data(), cnt * 2);
  return true;
}

std::string
Adapter::summary(const std::string& path, float scale) const
{
  return fmt("runtime LoRA '{}' -- {} modules at scale {}, rank <= {}{}{}",
             path, _modules, scale, _max_rank,
             _renamed > 0
                 ? fmt(", {} via the ai-toolkit/ComfyUI names", _renamed)()
                 : std::string(),
             _skipped > 0
                 ? fmt(", {} SKIPPED (shape mismatch)", _skipped)()
                 : std::string())();
}

bool
Adapter::file_touches(const std::string& path, const std::string& needle,
                      Rename rename)
{
  auto w = MetalLlamaWeights::open(path);
  if (!w.has_value()) { return false; }
  for (const std::string& n : w->tensor_names()) {
    if (n.find(needle) != std::string::npos) { return true; }
    if (rename == nullptr) { continue; }
    // The needle is a MODEL name, so the file's name has to be brought
    // into that spelling before it can be compared. The suffix goes
    // back on because the needles name a module AND its lora_ marker
    // (".ff.gate.lora_"), which is what keeps ".ff.gate" from matching
    // a base-weight name that happens to share the prefix.
    std::string suf;
    const std::string m = split_factor(n, &suf);
    if (m.empty()) { continue; }
    const std::string mapped = rename(m);
    if (!mapped.empty() && (mapped + suf).find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}


// ---- Applier -------------------------------------------------------

bool
Applier::init(MetalCompute* mc, bool allow_mma)
{
  _mc = mc;
  if (mc == nullptr) { return false; }
  _lib_gemm = mc->load_library("dense_gemm_bf16");
  _steel     = _lib_gemm.function("dense_gemm_t_bm64_f16");
  _steel_acc = _lib_gemm.function("dense_gemm_t_bm64_acc_f16");
  _mma = allow_mma && mc->supports_matrix_cores();
  if (_mma) {
    _lib_mma = mc->load_library("dense_gemm_mma_bf16");
    _a64  = _lib_mma.function("dense_gemm_mma_t_scaled_f16");
    _a128 = _lib_mma.function("dense_gemm_mma_t_n128_scaled_f16");
    _b128 = _lib_mma.function("dense_gemm_mma_t_n128_acc_f16");
    _b256 = _lib_mma.function("dense_gemm_mma_t_n128x256_acc_f16");
  }
  return valid();
}

bool
Applier::ensure_scratch(std::size_t elems)
{
  if (_mc == nullptr || elems == 0) { return false; }
  const std::size_t want = elems * 2;   // bf16
  if (_scratch.byte_size() >= want) { return true; }
  _scratch = _mc->make_shared_buffer(want);
  return !_scratch.empty();
}

// Which tile runs t = x A^T. Its N is the RANK, so this is purely a rank
// question: the 64-wide tile fits rank 64 without half of it hanging
// past N, and the 128-wide one takes over once there is a second tile's
// worth of rank to fill.
const metal_compute::ComputeFunction*
Applier::route_a_(int rank) const
{
  if (!_mma) { return nullptr; }
  const metal_compute::ComputeFunction* f = (rank <= 64) ? &_a64 : &_a128;
  if (!f->valid()) { f = &_a128; }
  return f->valid() ? f : nullptr;
}

const metal_compute::ComputeFunction*
Applier::route_b_(int rank) const
{
  if (!_mma || rank < 128) { return nullptr; }
  const metal_compute::ComputeFunction* f = (rank >= 256) ? &_b128 : &_b256;
  return f->valid() ? f : nullptr;
}

void
Applier::apply(metal_compute::ComputeEncoder& enc, const SharedBuffer& x,
               std::size_t x_off, const Factors& f, const SharedBuffer& y,
               std::size_t y_off, int m, int n, int k, float scale,
               int mma_min_m)
{
  if (f.empty() || !valid() || _scratch.empty()) { return; }
  if (scale == 0.0f) { return; }
  const int r = f.rank;
  const bool mma = _mma && m >= mma_min_m;

  // t = x A^T, into the scratch.
  const metal_compute::ComputeFunction* fa = mma ? route_a_(r) : nullptr;
  enc.set_function(fa != nullptr ? *fa : _steel);
  enc.set_buffer(0, x, x_off * 2);
  enc.set_buffer(1, f.a);
  enc.set_buffer(2, f.a);            // bias slot unused (has_bias = 0)
  enc.set_buffer(3, _scratch);
  enc.set_constant(4, k);
  enc.set_constant(5, r);
  enc.set_constant(6, m);
  enc.set_constant(7, 0);
  bool scaled = false;
  if (fa != nullptr) {
    enc.set_constant(8, scale);
    scaled = true;
    const int BN = (fa == &_a128) ? 128 : 64;
    const unsigned tw = (BN == 128) ? 256u : 128u;
    enc.dispatch({(unsigned)(((r + BN - 1) / BN) * tw),
                  (unsigned)((m + BN - 1) / BN), 1}, {tw, 1, 1});
  } else {
    enc.dispatch({(unsigned)(((r + 31) / 32) * 32),
                  (unsigned)(((m + 63) / 64) * 2), 2}, {32, 2, 2});
  }

  // y += t B^T. The accumulating tiles have no scale of their own, so
  // that route is legal only once the first GEMM took the strength.
  const metal_compute::ComputeFunction* fb =
      (mma && scaled) ? route_b_(r) : nullptr;
  enc.set_function(fb != nullptr ? *fb : _steel_acc);
  enc.set_buffer(0, _scratch);
  enc.set_buffer(1, f.b);
  enc.set_buffer(2, f.b);
  enc.set_buffer(3, y, y_off * 2);
  enc.set_constant(4, r);
  enc.set_constant(5, n);
  enc.set_constant(6, m);
  enc.set_constant(7, 0);
  enc.set_constant(8, scaled ? 1.0f : scale);
  if (fb != nullptr) {
    const int BN = (fb == &_b256) ? 256 : 128;
    enc.dispatch({(unsigned)(((n + BN - 1) / BN) * 256),
                  (unsigned)((m + 127) / 128), 1}, {256, 1, 1});
    return;
  }
  enc.dispatch({(unsigned)(((n + 31) / 32) * 32),
                (unsigned)(((m + 63) / 64) * 2), 2}, {32, 2, 2});
}

}  // namespace lora
}  // namespace genai
}  // namespace vpipe
