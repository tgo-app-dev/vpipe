#include "generative-models/flashvsr/flashvsr-lq-proj.h"

#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/mma-splitk.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <deque>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// Cache namespace for this class's derived tensors. A WeightSet is
// shared by everything reading one checkpoint, so the key has to say
// whose transform produced the bytes and every config input that
// changes them -- here the dtype and the im2col tap order.
constexpr const char* kKey = "flashvsr-lq/bf16/";

constexpr int kTapsT = 4;    // temporal taps of the (4,3,3) kernel
constexpr int kTaps  = 36;   // 4 * 3 * 3
constexpr int kCarry = FlashVsrLqProj::Config::kCarry;

std::uint16_t
to_bf16(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  // Round to nearest even, which is what every other bf16 conversion in
  // this tree does; truncation costs half a bit across 226M weights.
  const std::uint32_t r = (u >> 16) & 1u;
  u += 0x7fffu + r;
  return (std::uint16_t)(u >> 16);
}

// One element of a raw tensor as f32, whatever the checkpoint stores it
// as: F32 as is, BF16 widened, F16 by the plain bit-twiddle (no
// <arm_fp16.h> here).
inline float
elem_f32(const void* p, int kind, std::size_t i)
{
  float f = 0.0f;
  if (kind == 0) {
    std::memcpy(&f, (const std::uint8_t*)p + i * 4, 4);
    return f;
  }
  const std::uint16_t h = ((const std::uint16_t*)p)[i];
  std::uint32_t u = 0;
  if (kind == 1) {
    u = (std::uint32_t)h << 16;
  } else {
    const std::uint32_t sign = (std::uint32_t)(h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1fu;
    std::uint32_t man = h & 0x3ffu;
    if (exp == 0) {
      if (man == 0) {
        u = sign;
      } else {
        exp = 127 - 15 + 1;
        while ((man & 0x400u) == 0) { man <<= 1; --exp; }
        man &= 0x3ffu;
        u = sign | (exp << 23) | (man << 13);
      }
    } else if (exp == 31) {
      u = sign | 0x7f800000u | (man << 13);
    } else {
      u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
  }
  std::memcpy(&f, &u, 4);
  return f;
}

// A tensor straight to bf16 in a buffer of its own, element `d` taken
// from source element `src_of(d)` -- WITHOUT an f32 copy on the host.
//
// The copy is the point. This loader widened every weight to a host
// std::vector<float> (and the conv weights to a second, permuted one)
// before narrowing to bf16, and on this OS a freed large heap block stays
// in the process's footprint, dirty, until the process exits: MEASURED,
// loading this 549 MB projection left the footprint at 2746 MB, with the
// heap reporting 10 MB in use. On a 16 GB box that was the swap.
template <class Map>
SharedBuffer
read_bf16(WeightSet& ws, MetalCompute* mc, const std::string& nm,
          Map src_of, std::size_t* n_out = nullptr)
{
  const auto* info = ws.src().info(nm);
  if (info == nullptr) { return {}; }
  const int kind = info->dtype == "F32" ? 0 : info->dtype == "BF16" ? 1
                 : info->dtype == "F16" ? 2 : -1;
  if (kind < 0) { return {}; }
  std::size_t n = 1;
  for (const auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer raw = ws.read(nm, mc, WeightSet::Residency::Copied);
  if (raw.empty()) { return {}; }
  SharedBuffer out = mc->make_shared_buffer(n * 2);
  if (out.empty()) { return {}; }
  const void* p = raw.contents();
  auto* d = (std::uint16_t*)out.contents();
  for (std::size_t i = 0; i < n; ++i) {
    d[i] = to_bf16(elem_f32(p, kind, src_of(i)));
  }
  if (n_out != nullptr) { *n_out = n; }
  return out;
}

}  // namespace

// One (4,3,3) causal convolution, its weight already flattened to the
// tap order im2col_hwc_4x3x3_rep_tiled emits.
struct FlashVsrLqProj::Conv {
  SharedBuffer w;    // [cout, 36*cin] bf16
  SharedBuffer b;    // [cout] bf16
  int cin = 0, cout = 0;
  bool empty() const { return w.empty(); }
};

// The temporal carry of one convolution: its last two INPUT frames.
//
// Two slots addressed MODULO the running frame count rather than a
// shifted window, which is the same trick the Wan VAE's carry uses and
// for the same reason: a chunk here can be one frame, so shifting would
// mean a GPU round trip per chunk to move the older frame down.
struct FlashVsrLqProj::Carry {
  SharedBuffer buf;          // 2 slots of [hw * cin] bf16
  long long    total = 0;    // frames consumed so far
  std::size_t  hw    = 0;
  int          cin   = 0;
  bool live() const { return !buf.empty() && total > 0; }
};

// Scratch pool for one stream_forward, same shape as the Wan VAE's: a
// free list keyed on capacity, so the frames of one chunk reuse each
// other's buffers instead of allocating per frame.
struct FlashVsrLqProj::Ctx {
  ComputeEncoder* enc = nullptr;
  struct Slot { SharedBuffer buf; std::size_t cap; bool used; };
  std::deque<Slot> pool;
  bool alloc_ok = true;
  SharedBuffer col;          // im2col band
  std::size_t  col_cap = 0;  // ELEMENTS

  SharedBuffer&
  alloc(MetalCompute* mc, std::size_t elems)
  {
    const std::size_t bytes = elems * 2;
    for (auto& s : pool) {
      if (!s.used && !s.buf.empty() && s.cap >= bytes) {
        s.used = true;
        return s.buf;
      }
    }
    pool.push_back(Slot{mc->make_shared_buffer(bytes), bytes, true});
    if (pool.back().buf.empty()) { alloc_ok = false; }
    return pool.back().buf;
  }
};

FlashVsrLqProj::~FlashVsrLqProj() = default;

// ---------------------------------------------------------------- load

std::unique_ptr<FlashVsrLqProj>
FlashVsrLqProj::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                     std::string* err, const std::string& prefix)
{
  auto fail = [&](std::string m) -> std::unique_ptr<FlashVsrLqProj> {
    if (err != nullptr) { *err = std::move(m); }
    return nullptr;
  };
  if (ws == nullptr || mc == nullptr) { return fail("no weights or backend"); }

  auto m = std::unique_ptr<FlashVsrLqProj>(new FlashVsrLqProj());
  m->_mc = mc;
  m->_ws = std::move(ws);
  WeightSet& w = *m->_ws;

  // How many output projections the checkpoint ships decides how many
  // blocks the source reaches. Counted, never assumed: v1.1 ships one
  // and the module allows one per block, so assuming would be the
  // difference between conditioning one block and thirty.
  int n_out = 0;
  while (w.has(fmt("{}linear_layers.{}.weight", prefix, n_out)())) {
    ++n_out;
  }
  if (n_out == 0) {
    return fail(fmt("no {}linear_layers.* in checkpoint", prefix)());
  }
  m->_cfg.n_out_layers = n_out;

  m->_conv1 = m->load_conv_(w, prefix + "conv1", err);
  m->_conv2 = m->load_conv_(w, prefix + "conv2", err);
  if (m->_conv1 == nullptr || m->_conv2 == nullptr) {
    return fail(err != nullptr && !err->empty() ? *err
                                                : "lq_proj convs missing");
  }
  m->_cfg.hidden1 = m->_conv1->cout;
  m->_cfg.hidden2 = m->_conv2->cout;
  m->_cfg.in_channels = m->_conv1->cin / (m->_cfg.shuffle_h * m->_cfg.shuffle_w);

  m->_gamma1 = m->load_vec_(w, prefix + "norm1.gamma");
  m->_gamma2 = m->load_vec_(w, prefix + "norm2.gamma");
  if (m->_gamma1.empty() || m->_gamma2.empty()) {
    return fail("lq_proj norm gammas missing");
  }
  for (int i = 0; i < n_out; ++i) {
    SharedBuffer ow =
        m->load_mat_(w, fmt("{}linear_layers.{}", prefix, i)());
    SharedBuffer ob =
        m->load_vec_(w, fmt("{}linear_layers.{}.bias", prefix, i)());
    if (ow.empty()) { return fail("lq_proj output projection missing"); }
    m->_out_w.push_back(std::move(ow));
    m->_out_b.push_back(std::move(ob));
  }
  m->_cfg.out_dim = 1536;

  m->_lib_elt  = mc->load_library("llm_elementwise_bf16");
  m->_lib_gemm = mc->load_library("dense_gemm_bf16");
  m->_lib_rms  = mc->load_library("rms_norm_bf16");
  m->_fn_unshuffle = m->_lib_elt.function("pixel_unshuffle16_u8_hwc_f16");
  m->_fn_im2col    = m->_lib_elt.function("im2col_hwc_4x3x3_rep_tiled_f16");
  m->_fn_silu      = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_copy      = m->_lib_elt.function("copy_f16");
  m->_fn_rms       = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_gemm      = m->_lib_gemm.function("dense_gemm_bias_f16");
  // An unvalidated ComputeFunction is a SILENT no-op, so every one is
  // checked here rather than discovered as a zero activation later.
  if (!m->_fn_unshuffle.valid() || !m->_fn_im2col.valid() ||
      !m->_fn_silu.valid() || !m->_fn_copy.valid() || !m->_fn_rms.valid() ||
      !m->_fn_gemm.valid()) {
    return fail("flashvsr lq-proj kernels failed to validate");
  }

  // Matrix cores where the GPU has them, on the same terms as the
  // denoiser beside this: the tile and the split take the operands the
  // ALU kernel does, and the bias -- which no matmul2d tile applies --
  // is a second pass. A route that cannot add it is refused rather than
  // run biasless, since a dropped bias is a different answer.
  m->_splitk = std::make_unique<MmaSplitK>();
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_FVSR_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma_bf16");
    m->_fn_gemm_mma =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_gemm_mma_wide =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    m->_fn_bias_rows = m->_lib_elt.function("bias_add_rows_f16");
    m->_use_mma2 = m->_fn_gemm_mma.valid() &&
                   m->_fn_gemm_mma_wide.valid() && m->_fn_bias_rows.valid();
    if (m->_use_mma2) {
      m->_splitk->load(mc, m->_lib_dense_mma, m->_lib_elt);
    }
  }

  m->_carry1 = std::make_unique<Carry>();
  m->_carry2 = std::make_unique<Carry>();
  return m;
}

// A (4,3,3) conv weight [cout, cin, 4, 3, 3] flattened to
// [cout, 36*cin] with the inner index ((kt*3+ky)*3+kx)*cin + i, which is
// exactly the column order im2col_hwc_4x3x3_rep_tiled writes. The two
// orders have to be defined together or the GEMM silently pairs the
// wrong tap with the wrong channel.
std::unique_ptr<FlashVsrLqProj::Conv>
FlashVsrLqProj::load_conv_(WeightSet& ws, const std::string& nm,
                           std::string* err)
{
  const auto* info = ws.src().info(nm + ".weight");
  if (info == nullptr || info->shape.size() != 5) {
    if (err != nullptr) { *err = nm + ".weight missing or not 5-D"; }
    return nullptr;
  }
  const auto& sh = info->shape;
  const int cout = (int)sh[0], cin = (int)sh[1];
  const int kt = (int)sh[2], kh = (int)sh[3], kw = (int)sh[4];
  if (kt != kTapsT || kh != 3 || kw != 3) {
    if (err != nullptr) {
      *err = fmt("{}.weight is {}x{}x{}, expected 4x3x3", nm, kt, kh, kw)();
    }
    return nullptr;
  }
  auto c = std::make_unique<Conv>();
  c->cin = cin;
  c->cout = cout;
  c->w = ws.derived(kKey + std::string("conv|") + nm, [&]() -> SharedBuffer {
    // Destination [cout][tap][cin], source [cout][cin][t][ky][kx]: the
    // tap order im2col_hwc_4x3x3_rep_tiled emits.
    const std::size_t taps = (std::size_t)kTaps, ci = (std::size_t)cin;
    return read_bf16(ws, _mc, nm + ".weight", [&](std::size_t di) {
      const std::size_t i = di % ci;
      const std::size_t tap = (di / ci) % taps;
      const std::size_t o = di / (ci * taps);
      return (o * ci + i) * taps + tap;
    });
  });
  if (c->w.empty()) {
    if (err != nullptr) { *err = nm + ".weight could not be materialised"; }
    return nullptr;
  }
  c->b = load_vec_(ws, nm + ".bias");
  return c;
}

SharedBuffer
FlashVsrLqProj::load_vec_(WeightSet& ws, const std::string& nm)
{
  if (!ws.has(nm)) { return {}; }
  return ws.derived(kKey + std::string("vec|") + nm, [&]() -> SharedBuffer {
    return read_bf16(ws, _mc, nm, [](std::size_t i) { return i; });
  });
}

SharedBuffer
FlashVsrLqProj::load_mat_(WeightSet& ws, const std::string& nm)
{
  if (!ws.has(nm + ".weight")) { return {}; }
  return ws.derived(kKey + std::string("mat|") + nm, [&]() -> SharedBuffer {
    return read_bf16(ws, _mc, nm + ".weight", [](std::size_t i) { return i; });
  });
}

std::size_t
FlashVsrLqProj::weight_bytes(const std::string& source,
                             const std::string& prefix)
{
  auto ws = WeightSet::open(source, nullptr);
  if (ws == nullptr) { return 0; }
  std::size_t n = 0;
  for (const std::string& nm : ws->src().tensor_names()) {
    if (nm.compare(0, prefix.size(), prefix) != 0) { continue; }
    const auto* info = ws->src().info(nm);
    if (info == nullptr) { continue; }
    std::size_t e = 1;
    for (const auto d : info->shape) { e *= (std::size_t)d; }
    n += e * 2;                                   // held as bf16
  }
  return n;
}

std::size_t
FlashVsrLqProj::working_bytes(const std::string& source,
                              const std::string& prefix, int height,
                              int width, int frames)
{
  if (height <= 0 || width <= 0 || frames <= 0) { return 0; }
  auto ws = WeightSet::open(source, nullptr);
  if (ws == nullptr) { return 0; }
  const auto* c1 = ws->src().info(prefix + "conv1.weight");
  const auto* c2 = ws->src().info(prefix + "conv2.weight");
  const auto* lo = ws->src().info(prefix + "linear_layers.0.weight");
  if (c1 == nullptr || c2 == nullptr || lo == nullptr ||
      c1->shape.size() != 5 || c2->shape.size() != 5 ||
      lo->shape.size() != 2) {
    return 0;
  }
  using U = std::size_t;
  const Config cfg;
  const U hw = (U)(height / cfg.shuffle_h) * (U)(width / cfg.shuffle_w);
  const U C0 = (U)c1->shape[1], C1 = (U)c1->shape[0];
  const U C2 = (U)c2->shape[0], out = (U)lo->shape[0];
  // The call pattern stream_forward is driven with: four source frames a
  // call, which conv1 (stride 2 over the carry) turns into two and conv2
  // into one row frame.
  constexpr U kIn = 4, kOut1 = 2, kOut2 = 1;
  const U widest = (U)kTaps * std::max(C0, C1);
  const U band = std::min(hw * widest, (U)(192u << 20) / 2) * 2;
  const U src = kIn * 3 * (U)height * (U)width;          // u8 upload
  const U unshuffled = kIn * hw * C0 * 2;
  const U conv1 = 2 * kOut1 * hw * C1 * 2;               // and its norm
  const U conv2 = 2 * kOut2 * hw * C2 * 2;
  const U carries = (U)kCarry * hw * (C0 + C1) * 2;
  // The clip's row frames, one per four source frames past the warmup --
  // the latent's own count -- held as they come and then copied into the
  // beat.
  const int chunks = (frames - 1) / 8 - 2;
  const U row_frames = chunks > 0 ? (U)(2 * chunks + 4) : 0;
  const U rows = 2 * row_frames * hw * out * 2;
  return band + src + unshuffled + conv1 + conv2 + carries + rows;
}

void
FlashVsrLqProj::reset()
{
  if (_carry1 != nullptr) { _carry1->total = 0; }
  if (_carry2 != nullptr) { _carry2->total = 0; }
  _opened = false;
}

// ------------------------------------------------------------- forward

namespace {

// Where one frame of a [frames, hw, C] activation starts, in ELEMENTS.
inline std::size_t
frame_off(int f, std::size_t hw, int C)
{
  return (std::size_t)f * hw * (std::size_t)C;
}

// The frame offsets of a plain contiguous [frames, hw, C] buffer.
std::vector<std::size_t>
contiguous_offs(int frames, std::size_t hw, int C)
{
  std::vector<std::size_t> v((std::size_t)std::max(frames, 0));
  for (int j = 0; j < frames; ++j) { v[(std::size_t)j] = frame_off(j, hw, C); }
  return v;
}

}  // namespace

void
FlashVsrLqProj::gemm_bias_(Ctx& cx, const SharedBuffer& x,
                           const SharedBuffer& w, const SharedBuffer& b,
                           const SharedBuffer& y, int M, int N, int K,
                           std::size_t y_off, std::size_t x_off)
{
  ComputeEncoder& enc = *cx.enc;
  if (_use_mma2 && M >= 64 && N >= 16) {
    // The deep columns take the split planes when the split has a
    // compiled chunk for this K; the rest take one tile, banded so no
    // operand passes the tiles' 32-bit addressing (shared/mma-tile.h).
    if (!_splitk->encode(_mc, enc, x, w, y, K, N, M, x_off, y_off)) {
      const bool wide = mma_use_wide_tile(N, K);
      const metal_compute::ComputeFunction& fn =
          wide ? _fn_gemm_mma_wide : _fn_gemm_mma;
      const int RN = wide ? 256 : 128;
      const int band = mma_row_band(N, K);
      for (int r0 = 0; r0 < M; r0 += band) {
        const int rows = std::min(band, M - r0);
        enc.set_function(fn);
        enc.set_buffer(0, x, (x_off + (std::size_t)r0 * K) * 2);
        enc.set_buffer(1, w);
        enc.set_buffer(2, w);            // unread; the slot must be bound
        enc.set_buffer(3, y, (y_off + (std::size_t)r0 * N) * 2);
        enc.set_constant(4, K);
        enc.set_constant(5, N);
        enc.set_constant(6, rows);
        enc.set_constant(7, 0);
        enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                      (unsigned)((rows + 127) / 128), 1}, {256, 1, 1});
      }
    }
    if (!b.empty()) {
      enc.set_function(_fn_bias_rows);
      enc.set_buffer(0, y, y_off * 2);
      enc.set_buffer(1, b);
      enc.set_constant(2, N);
      const unsigned total = (unsigned)((std::size_t)M * N);
      enc.set_constant(3, total);
      enc.dispatch({(unsigned)N, (unsigned)M, 1}, {32, 8, 1});
    }
    return;
  }
  enc.set_function(_fn_gemm);
  enc.set_buffer(0, x, x_off * 2);
  enc.set_buffer(1, w);
  enc.set_buffer(2, b.empty() ? w : b);
  enc.set_buffer(3, y, y_off * 2);
  enc.set_constant(4, M);
  enc.set_constant(5, N);
  enc.set_constant(6, K);
  enc.set_constant(7, b.empty() ? 0 : 1);
  enc.dispatch({(unsigned)(((N + 15) / 16) * 16),
                (unsigned)(((M + 15) / 16) * 16), 1}, {16, 16, 1});
}

void
FlashVsrLqProj::conv_frame_(Ctx& cx, const Conv& c,
                            const SharedBuffer* const taps[4],
                            const std::size_t tap_off[4],
                            const SharedBuffer& out, std::size_t out_off,
                            int OH, int OW)
{
  ComputeEncoder& enc = *cx.enc;
  const std::size_t ohw = (std::size_t)OH * OW;
  const std::size_t per_row = (std::size_t)kTaps * c.cin;
  // Band the gather: at 36 taps over 2048 channels one row is 73728
  // elements, so the whole column matrix is hundreds of MB at any real
  // resolution and exists only a band at a time.
  std::size_t rows = (per_row > 0) ? (cx.col_cap / per_row) : ohw;
  rows = std::min(std::max<std::size_t>(rows, 1), ohw);
  for (std::size_t r0 = 0; r0 < ohw; r0 += rows) {
    const int nr = (int)std::min(rows, ohw - r0);
    enc.set_function(_fn_im2col);
    for (int k = 0; k < kTapsT; ++k) {
      enc.set_buffer((unsigned)k, *taps[k], tap_off[k] * 2);
    }
    enc.set_buffer(4, cx.col);
    enc.set_constant(5, OH);
    enc.set_constant(6, OW);
    enc.set_constant(7, c.cin);
    enc.set_constant(8, (int)r0);
    enc.set_constant(9, nr);
    enc.dispatch({(unsigned)per_row, (unsigned)nr, 1}, {64, 1, 1});
    gemm_bias_(cx, cx.col, c.w, c.b, out, nr, c.cout, (int)per_row,
               out_off + r0 * (std::size_t)c.cout);
  }
}

metal_compute::SharedBuffer&
FlashVsrLqProj::norm_silu_(Ctx& cx, const SharedBuffer& x, std::size_t rows,
                           int C, const SharedBuffer& gamma)
{
  ComputeEncoder& enc = *cx.enc;
  SharedBuffer& y = cx.alloc(_mc, rows * (std::size_t)C);
  if (!cx.alloc_ok) { return y; }
  // The reference's norm is `F.normalize(x, dim=channels) * sqrt(C) *
  // gamma`, which is the same function as x * rsqrt(mean(x^2) + eps) *
  // gamma with eps at F.normalize's own 1e-12 clamp -- divided by C
  // because that clamp is on the SUM of squares and this kernel's eps
  // sits beside their MEAN.
  const float eps = 1e-12f / (float)C;
  enc.set_function(_fn_rms);
  enc.set_buffer(0, x);
  enc.set_buffer(1, gamma);
  enc.set_buffer(2, y);
  enc.set_constant(3, C);
  enc.set_constant(4, eps);
  enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});

  const std::size_t n = rows * (std::size_t)C;
  enc.set_function(_fn_silu);
  enc.set_buffer(0, y);
  enc.set_buffer(1, y);
  enc.set_buffer(2, y);
  enc.set_constant(3, (int)n);
  enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  return y;
}

void
FlashVsrLqProj::save_carry_(Ctx& cx, Carry& carry, const SharedBuffer& src,
                            const std::vector<std::size_t>& frame_offs,
                            std::size_t hw, int cin)
{
  const int m = (int)frame_offs.size();
  if (m <= 0) { return; }
  const std::size_t frame_el = hw * (std::size_t)cin;
  if (carry.buf.empty() || carry.hw != hw || carry.cin != cin) {
    carry.buf = _mc->make_shared_buffer((std::size_t)kCarry * frame_el * 2);
    if (carry.buf.empty()) { cx.alloc_ok = false; return; }
    carry.hw = hw;
    carry.cin = cin;
    carry.total = 0;
  }
  // Only the last two frames can still be read, and each goes to the
  // slot its SEQUENCE index selects, so nothing moves when a one-frame
  // chunk lands next to a two-frame history. GPU-side copies: at this
  // point src exists only on the GPU timeline.
  ComputeEncoder& enc = *cx.enc;
  const int first = std::max(0, m - kCarry);
  for (int j = first; j < m; ++j) {
    const std::size_t slot =
        (std::size_t)((carry.total + j) % (long long)kCarry);
    enc.set_function(_fn_copy);
    enc.set_buffer(0, src, frame_offs[(std::size_t)j] * 2);
    enc.set_buffer(1, carry.buf);
    enc.set_constant(2, (int)(slot * frame_el));
    enc.set_constant(3, (int)frame_el);
    enc.dispatch({(unsigned)frame_el, 1, 1}, {256, 1, 1});
  }
  carry.total += m;
}

bool
FlashVsrLqProj::stream_forward(const std::uint8_t* lq, int n, int height,
                               int width, std::vector<SharedBuffer>* out,
                               int* out_row_frames, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (out_row_frames != nullptr) { *out_row_frames = 0; }
  if (lq == nullptr || n <= 0) { return fail("empty source chunk"); }
  if (_conv1 == nullptr || _conv2 == nullptr) { return fail("not loaded"); }
  const int sh = _cfg.shuffle_h, sw = _cfg.shuffle_w;
  if (height <= 0 || width <= 0 || height % sh != 0 || width % sw != 0) {
    return fail(fmt("source {}x{} is not a multiple of {}x{}", width, height,
                    sw, sh)());
  }
  const int OH = height / sh, OW = width / sw;
  const std::size_t hw = (std::size_t)OH * OW;
  const int C0 = _conv1->cin;

  MetalCompute* mc = _mc;
  CommandStream stream = mc->make_command_stream();
  Ctx cx;

  const std::size_t widest =
      (std::size_t)kTaps * (std::size_t)std::max(_conv1->cin, _conv2->cin);
  const std::size_t full_band = hw * widest;
  std::size_t col_cap = std::min(full_band, (std::size_t)(192u << 20) / 2);
  col_cap = std::max(col_cap, widest);          // at least one row
  cx.col = mc->make_shared_buffer(col_cap * 2);
  if (cx.col.empty()) { return fail("im2col band allocation failed"); }
  cx.col_cap = col_cap;

  // The source frames, uploaded once as u8 -- the only full copy of this
  // chunk anywhere in the call.
  const std::size_t frame_px = (std::size_t)3 * height * width;
  SharedBuffer src = mc->make_shared_buffer(frame_px * (std::size_t)n);
  if (src.empty()) { return fail("source upload allocation failed"); }
  std::memcpy(src.contents(), lq, frame_px * (std::size_t)n);

  int out_frames = 0;
  std::vector<SharedBuffer> produced;
  {
    ComputeEncoder enc = stream.begin_compute();
    cx.enc = &enc;

    SharedBuffer& u = cx.alloc(mc, (std::size_t)n * hw * C0);
    if (!cx.alloc_ok) { return fail("unshuffle allocation failed"); }
    for (int f = 0; f < n; ++f) {
      enc.set_function(_fn_unshuffle);
      enc.set_buffer(0, src, frame_px * (std::size_t)f);
      enc.set_buffer(1, u, frame_off(f, hw, C0) * 2);
      enc.set_constant(2, height);
      enc.set_constant(3, width);
      enc.dispatch({(unsigned)C0, (unsigned)hw, 1}, {64, 1, 1});
    }

    // The opening call prepends three copies of the first frame. That is
    // a MAPPING, not a copy: logical frame j reads physical max(0, j-3).
    const int lead = _opened ? 0 : 3;
    const int m = lead + n;
    auto logical = [&](int j) {
      return frame_off(_opened ? j : std::max(0, j - lead), hw, C0);
    };

    // ---- conv1, stride 2 in time over the padded sequence.
    const int out1 =
        (m + kCarry >= kTapsT) ? ((m + kCarry - kTapsT) / 2 + 1) : 0;
    if (out1 <= 0) { return fail("source chunk too short to convolve"); }
    SharedBuffer* a = &cx.alloc(mc, (std::size_t)out1 * hw * _conv1->cout);
    if (!cx.alloc_ok) { return fail("conv1 output allocation failed"); }
    for (int o = 0; o < out1; ++o) {
      const SharedBuffer* taps[kTapsT];
      std::size_t offs[kTapsT];
      for (int k = 0; k < kTapsT; ++k) {
        const int p = 2 * o + k;              // index into the padded seq
        if (_carry1->live() && p < kCarry) {
          taps[k] = &_carry1->buf;
          offs[k] = (std::size_t)((_carry1->total - kCarry + p) %
                                  (long long)kCarry) * hw * (std::size_t)C0;
        } else if (_carry1->live()) {
          taps[k] = &u;
          offs[k] = logical(p - kCarry);
        } else {
          // No carry: the left edge REPLICATES the opening frame, which
          // is what binding the same view twice says.
          taps[k] = &u;
          offs[k] = logical(std::max(0, p - kCarry));
        }
      }
      conv_frame_(cx, *_conv1, taps, offs, *a,
                  frame_off(o, hw, _conv1->cout), OH, OW);
    }

    // carry1 takes the last two frames of conv1's INPUT, and it is read
    // above before it is written here. That order is the whole
    // difference between this module and the reference's other, buffered
    // spelling of it.
    {
      std::vector<std::size_t> offs((std::size_t)m);
      for (int j = 0; j < m; ++j) { offs[(std::size_t)j] = logical(j); }
      save_carry_(cx, *_carry1, u, offs, hw, C0);
    }

    a = &norm_silu_(cx, *a, (std::size_t)out1 * hw, _conv1->cout, _gamma1);
    if (!cx.alloc_ok) { return fail("norm scratch allocation failed"); }

    if (!_opened) {
      // The opening call seeds conv2's carry from conv1's output and
      // produces no rows at all.
      save_carry_(cx, *_carry2, *a,
                  contiguous_offs(out1, hw, _conv1->cout), hw, _conv1->cout);
      _opened = true;
    } else {
      const int m2 = out1;
      const int out2 =
          (m2 + kCarry >= kTapsT) ? ((m2 + kCarry - kTapsT) / 2 + 1) : 0;
      if (out2 > 0) {
        SharedBuffer* b =
            &cx.alloc(mc, (std::size_t)out2 * hw * _conv2->cout);
        if (!cx.alloc_ok) { return fail("conv2 output allocation failed"); }
        for (int o = 0; o < out2; ++o) {
          const SharedBuffer* taps[kTapsT];
          std::size_t offs[kTapsT];
          for (int k = 0; k < kTapsT; ++k) {
            const int p = 2 * o + k;
            if (_carry2->live() && p < kCarry) {
              taps[k] = &_carry2->buf;
              offs[k] = (std::size_t)((_carry2->total - kCarry + p) %
                                      (long long)kCarry) * hw *
                        (std::size_t)_conv1->cout;
            } else if (_carry2->live()) {
              taps[k] = a;
              offs[k] = frame_off(p - kCarry, hw, _conv1->cout);
            } else {
              taps[k] = a;
              offs[k] = frame_off(std::max(0, p - kCarry), hw, _conv1->cout);
            }
          }
          conv_frame_(cx, *_conv2, taps, offs, *b,
                      frame_off(o, hw, _conv2->cout), OH, OW);
        }
        // carry2 likewise reads before it writes.
        save_carry_(cx, *_carry2, *a,
                    contiguous_offs(out1, hw, _conv1->cout), hw,
                    _conv1->cout);

        b = &norm_silu_(cx, *b, (std::size_t)out2 * hw, _conv2->cout,
                        _gamma2);
        if (!cx.alloc_ok) { return fail("norm scratch allocation failed"); }

        // The output projections. Rows run (frame, row, column) with the
        // channel fastest, which is the layout the denoiser adds into
        // its residual stream.
        const std::size_t n_rows = (std::size_t)out2 * hw;
        for (std::size_t i = 0; i < _out_w.size(); ++i) {
          // NOT from the pool: these leave the call as the caller's, so
          // they must outlive the scratch that produced them.
          SharedBuffer r =
              mc->make_shared_buffer(n_rows * (std::size_t)_cfg.out_dim * 2);
          if (r.empty()) { return fail("row allocation failed"); }
          gemm_bias_(cx, *b, _out_w[i], _out_b[i], r, (int)n_rows,
                     _cfg.out_dim, _conv2->cout);
          produced.push_back(std::move(r));
        }
        out_frames = out2;
      } else {
        save_carry_(cx, *_carry2, *a,
                    contiguous_offs(out1, hw, _conv1->cout), hw,
                    _conv1->cout);
      }
    }
    enc.end();
  }

  std::string gpu_err;
  if (!stream.commit().wait_ok(&gpu_err)) {
    return fail("lq-proj GPU work failed: " + gpu_err);
  }
  if (!cx.alloc_ok) { return fail("lq-proj scratch allocation failed"); }
  if (out != nullptr) {
    for (auto& b : produced) { out->push_back(std::move(b)); }
  }
  if (out_row_frames != nullptr) { *out_row_frames = out_frames; }
  return true;
}

std::uint64_t
FlashVsrLqProj::resident_bytes() const noexcept
{
  std::uint64_t n = 0;
  auto add = [&n](const SharedBuffer& b) { n += b.byte_size(); };
  if (_conv1 != nullptr) { add(_conv1->w); add(_conv1->b); }
  if (_conv2 != nullptr) { add(_conv2->w); add(_conv2->b); }
  add(_gamma1);
  add(_gamma2);
  for (const auto& b : _out_w) { add(b); }
  for (const auto& b : _out_b) { add(b); }
  if (_carry1 != nullptr) { add(_carry1->buf); }
  if (_carry2 != nullptr) { add(_carry2->buf); }
  return n;
}

}  // namespace genai
}  // namespace vpipe
