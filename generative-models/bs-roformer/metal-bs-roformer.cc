#include "generative-models/bs-roformer/metal-bs-roformer.h"

#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/shared/stft.h"
#include "generative-models/weight-set.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

namespace {

// Cache namespace for this class's derived tensors: the transforms here
// are a dtype change to f16 (or f32 for the band split) and the band
// concatenations, none of which depend on anything but the checkpoint.
constexpr const char* kKey = "bs-roformer/v1/";

// F.normalize's 1e-12 floor on the norm, restated as rms_norm_f16's
// epsilon on the mean square: small enough to change nothing measurable
// and enough that an all-zero row stays zero instead of 0 / 0.
constexpr float kNormEps = 1e-24f;

// C++ mirror of the steel attention's params (vendored MLX
// steel/attn/params.h), as the other callers here keep it.
struct SteelAttnParams {
  int B, H, D;
  int qL, kL;
  int gqa_factor;
  float scale;
  int NQ, NK;
  int NQ_aligned, NK_aligned;
  int qL_rem, kL_rem, qL_off;
  std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};

std::size_t
numel_(const std::vector<std::int64_t>& s)
{
  std::size_t n = 1;
  for (const auto d : s) { n *= (std::size_t)d; }
  return n;
}

// One element of a raw tensor as f32: F32 as is, BF16 widened, F16 by
// the compiler's _Float16.
inline float
elem_f32_(const void* p, int kind, std::size_t i)
{
  if (kind == 0) {
    float f;
    std::memcpy(&f, (const std::uint8_t*)p + i * 4, 4);
    return f;
  }
  const std::uint16_t h = ((const std::uint16_t*)p)[i];
  if (kind == 1) {
    const std::uint32_t u = (std::uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  }
  _Float16 v;
  std::memcpy(&v, &h, 2);
  return (float)v;
}

// A tensor read whole as f32 on the host, with its shape. Consumed by
// the band concatenations below, so read() -- uncached -- not tensor().
bool
read_f32_(WeightSet& ws, metal_compute::MetalCompute* mc,
          const std::string& nm, std::vector<float>* out,
          std::vector<std::int64_t>* shape, std::string* err)
{
  const auto* info = ws.src().info(nm);
  if (info == nullptr) {
    if (err != nullptr) { *err = "the checkpoint has no " + nm; }
    return false;
  }
  const int kind = info->dtype == "F32" ? 0 : info->dtype == "BF16" ? 1
                 : info->dtype == "F16" ? 2 : -1;
  if (kind < 0) {
    if (err != nullptr) { *err = nm + " is " + info->dtype + ", not a float"; }
    return false;
  }
  const std::size_t n = numel_(info->shape);
  SharedBuffer raw = ws.read(nm, mc, WeightSet::Residency::Copied);
  if (raw.empty()) {
    if (err != nullptr) { *err = nm + " could not be read"; }
    return false;
  }
  out->resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    (*out)[i] = elem_f32_(raw.contents(), kind, i);
  }
  if (shape != nullptr) { *shape = info->shape; }
  return true;
}

SharedBuffer
to_f16_(metal_compute::MetalCompute* mc, const std::vector<float>& v)
{
  SharedBuffer b = mc->make_shared_buffer(std::max<std::size_t>(v.size(), 1)
                                          * 2);
  if (b.empty()) { return b; }
  auto* d = (_Float16*)b.contents();
  for (std::size_t i = 0; i < v.size(); ++i) { d[i] = (_Float16)v[i]; }
  return b;
}

SharedBuffer
to_f32_(metal_compute::MetalCompute* mc, const std::vector<float>& v)
{
  SharedBuffer b = mc->make_shared_buffer(std::max<std::size_t>(v.size(), 1)
                                          * 4);
  if (b.empty()) { return b; }
  std::memcpy(b.contents(), v.data(), v.size() * 4);
  return b;
}

bool
shape_is_(const std::vector<std::int64_t>& s,
          std::initializer_list<std::int64_t> want)
{
  return s.size() == want.size() &&
         std::equal(s.begin(), s.end(), want.begin());
}

double
ms_since_(std::chrono::steady_clock::time_point t0)
{
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

MetalBsRoformer::~MetalBsRoformer() = default;

std::unique_ptr<MetalBsRoformer>
MetalBsRoformer::load(std::shared_ptr<WeightSet> ws,
                      metal_compute::MetalCompute* mc, const Config& cfg,
                      std::string* err)
{
  auto fail = [&](const std::string& what) {
    if (err != nullptr) { *err = what; }
    return std::unique_ptr<MetalBsRoformer>();
  };
  if (mc == nullptr || !mc->valid()) { return fail("no metal-compute"); }
  if (ws == nullptr) { return fail("no checkpoint"); }
  // What the kernels below were written for. Every config published so
  // far fits; one that does not is refused here rather than run wrong.
  if (cfg.dim % 32 != 0 || cfg.dim > 1024) {
    return fail("dim " + std::to_string(cfg.dim) +
                " must be a multiple of 32, at most 1024");
  }
  if (cfg.dim_head != 64) {
    return fail("dim_head " + std::to_string(cfg.dim_head) +
                ": the attention kernels here take 64");
  }
  if (cfg.heads > 16) {
    return fail("heads " + std::to_string(cfg.heads) + ": at most 16");
  }
  for (int f : cfg.freqs_per_bands) {
    if (f * cfg.channels() * 2 > 1024) {
      return fail("a band of " + std::to_string(f) +
                  " bins is wider than the band split takes (1024 "
                  "features)");
    }
  }

  auto m = std::unique_ptr<MetalBsRoformer>(new MetalBsRoformer());
  m->_cfg = cfg;
  m->_mc = mc;
  m->_ws = std::move(ws);

  m->_lib_bsr = mc->load_library("band_split_roformer");
  m->_lib_gemm = mc->load_library("dense_gemm");
  m->_lib_elt = mc->load_library("llm_elementwise");
  m->_lib_rms = mc->load_library("rms_norm");
  m->_fn_split = m->_lib_bsr.function("bsr_band_split_f16");
  m->_fn_rope = m->_lib_bsr.function("bsr_rope_rows_f16");
  m->_fn_gate = m->_lib_bsr.function("bsr_head_gate_f16");
  m->_fn_bias_tanh = m->_lib_bsr.function("bsr_bias_tanh_f16");
  m->_fn_glu = m->_lib_bsr.function("bsr_band_glu_f16");
  m->_fn_bias_gelu = m->_lib_bsr.function("bsr_bias_gelu_f16");
  m->_fn_bias_res = m->_lib_bsr.function("bsr_bias_residual_f16");
  m->_fn_rms = m->_lib_rms.function("rms_norm_f16");
  m->_fn_gelu = m->_lib_elt.function("gelu_erf_f16");
  m->_fn_residual = m->_lib_elt.function("residual_add_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_bias_add = m->_lib_elt.function("bias_add_rows_f16");
  m->_fn_gemm_t = m->_lib_gemm.function("dense_gemm_t_f16");
  m->_fn_gemm_64 = m->_lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  if (!m->_fn_split.valid() || !m->_fn_rope.valid() ||
      !m->_fn_gate.valid() || !m->_fn_bias_tanh.valid() ||
      !m->_fn_glu.valid() || !m->_fn_bias_gelu.valid() ||
      !m->_fn_bias_res.valid() || !m->_fn_rms.valid() || !m->_fn_gelu.valid() ||
      !m->_fn_residual.valid() || !m->_fn_transpose.valid() ||
      !m->_fn_bias_add.valid() || !m->_fn_gemm_t.valid() ||
      !m->_fn_gemm_64.valid()) {
    return fail("a kernel did not load (band_split_roformer, dense_gemm, "
                "llm_elementwise or rms_norm)");
  }
  // matmul2d where the GPU has matrix cores; VPIPE_BS_ROFORMER_NO_MMA2=1
  // keeps the steel tiles (the A/B).
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_BS_ROFORMER_NO_MMA2") == nullptr) {
    m->_lib_mma = mc->load_library("dense_gemm_mma");
    m->_fn_mma = m->_lib_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_mma_wide = m->_lib_mma.function("dense_gemm_mma_t_n128x256_f16");
    m->_use_mma2 = m->_fn_mma.valid() && m->_fn_mma_wide.valid();
  }
  // The attention: NAX on matrix cores (VPIPE_BS_ROFORMER_NO_NAX_ATTN=1
  // reverts), the ALU steel kernel otherwise.
  m->_attn_nax = mc->supports_matrix_cores() &&
                 std::getenv("VPIPE_BS_ROFORMER_NO_NAX_ATTN") == nullptr;
  m->_lib_attn = mc->load_library(m->_attn_nax ? "attn_steel_nax"
                                               : "attn_steel");
  if (!m->_lib_attn.valid() && m->_attn_nax) {
    m->_attn_nax = false;
    m->_lib_attn = mc->load_library("attn_steel");
  }
  if (!m->_lib_attn.valid()) { return fail("attn_steel did not load"); }

  if (!m->load_weights_(err)) { return nullptr; }
  if (!m->alloc_scratch_(err)) { return nullptr; }
  return m;
}

bool
MetalBsRoformer::load_weights_(std::string* err)
{
  const Config& c = _cfg;
  WeightSet& ws = *_ws;
  const int D = c.dim, nb = c.bands(), H = c.heads, Dh = c.dim_head;
  const int inner = c.inner(), S = c.channels(), Fin = c.frame_features();
  auto fail = [&](const std::string& what) {
    if (err != nullptr) { *err = what; }
    return false;
  };
  // Every tensor this model keeps comes through derived(), keyed by its
  // checkpoint name and transform, so two models over one checkpoint
  // share them.
  auto vec16 = [&](const std::string& nm, std::int64_t n,
                   SharedBuffer* out) {
    std::string e;
    *out = ws.derived(std::string(kKey) + "f16|" + nm, [&]() {
      std::vector<float> v;
      std::vector<std::int64_t> s;
      if (!read_f32_(ws, _mc, nm, &v, &s, &e)) { return SharedBuffer(); }
      if ((std::int64_t)v.size() != n) {
        e = nm + " holds " + std::to_string(v.size()) + " values, not " +
            std::to_string(n);
        return SharedBuffer();
      }
      return to_f16_(_mc, v);
    });
    if (out->empty()) { return fail(e.empty() ? nm + " failed" : e); }
    _weight_bytes += (std::size_t)n * 2;
    return true;
  };
  auto mat16 = [&](const std::string& nm, std::int64_t rows,
                   std::int64_t cols, SharedBuffer* out) {
    std::string e;
    *out = ws.derived(std::string(kKey) + "f16|" + nm, [&]() {
      std::vector<float> v;
      std::vector<std::int64_t> s;
      if (!read_f32_(ws, _mc, nm, &v, &s, &e)) { return SharedBuffer(); }
      if (!shape_is_(s, {rows, cols})) {
        e = nm + " is not [" + std::to_string(rows) + ", " +
            std::to_string(cols) + "]";
        return SharedBuffer();
      }
      return to_f16_(_mc, v);
    });
    if (out->empty()) { return fail(e.empty() ? nm + " failed" : e); }
    _weight_bytes += (std::size_t)(rows * cols) * 2;
    return true;
  };

  // ---- the band split: every band's norm and Linear, stacked ----
  _band_off.assign((std::size_t)nb, 0);
  _band_len.assign((std::size_t)nb, 0);
  {
    int off = 0;
    std::vector<int> tab((std::size_t)nb * 2);
    for (int b = 0; b < nb; ++b) {
      const int len = c.freqs_per_bands[(std::size_t)b] * S * 2;
      _band_off[(std::size_t)b] = off;
      _band_len[(std::size_t)b] = len;
      tab[(std::size_t)b * 2] = off;
      tab[(std::size_t)b * 2 + 1] = len;
      off += len;
    }
    _band_tab = _mc->make_shared_buffer(tab.size() * 4);
    if (_band_tab.empty()) { return fail("out of memory"); }
    std::memcpy(_band_tab.contents(), tab.data(), tab.size() * 4);
  }
  {
    std::string e;
    _band_gamma = ws.derived(std::string(kKey) + "band-gamma", [&]() {
      std::vector<float> all((std::size_t)Fin);
      for (int b = 0; b < nb; ++b) {
        const std::string nm = "band_split.to_features." +
                               std::to_string(b) + ".0.gamma";
        std::vector<float> v;
        std::vector<std::int64_t> s;
        if (!read_f32_(ws, _mc, nm, &v, &s, &e)) { return SharedBuffer(); }
        if ((int)v.size() != _band_len[(std::size_t)b]) {
          e = nm + " does not match band " + std::to_string(b) + "'s " +
              std::to_string(_band_len[(std::size_t)b]) + " features";
          return SharedBuffer();
        }
        std::copy(v.begin(), v.end(),
                  all.begin() + _band_off[(std::size_t)b]);
      }
      return to_f32_(_mc, all);
    });
    if (_band_gamma.empty()) { return fail(e); }
    // W_b^T stacked: row (off_b + k) holds input feature k of band b.
    _band_wt = ws.derived(std::string(kKey) + "band-wt", [&]() {
      std::vector<float> all((std::size_t)Fin * D);
      for (int b = 0; b < nb; ++b) {
        const std::string nm = "band_split.to_features." +
                               std::to_string(b) + ".1.weight";
        const int len = _band_len[(std::size_t)b];
        std::vector<float> v;
        std::vector<std::int64_t> s;
        if (!read_f32_(ws, _mc, nm, &v, &s, &e)) { return SharedBuffer(); }
        if (!shape_is_(s, {D, len})) {
          e = nm + " is not [" + std::to_string(D) + ", " +
              std::to_string(len) + "]";
          return SharedBuffer();
        }
        for (int o = 0; o < D; ++o) {
          for (int k = 0; k < len; ++k) {
            all[(std::size_t)(_band_off[(std::size_t)b] + k) * D + o] =
                v[(std::size_t)o * len + k];
          }
        }
      }
      return to_f32_(_mc, all);
    });
    if (_band_wt.empty()) { return fail(e); }
    _band_bias = ws.derived(std::string(kKey) + "band-bias", [&]() {
      std::vector<float> all((std::size_t)nb * D);
      for (int b = 0; b < nb; ++b) {
        const std::string nm = "band_split.to_features." +
                               std::to_string(b) + ".1.bias";
        std::vector<float> v;
        if (!read_f32_(ws, _mc, nm, &v, nullptr, &e)) {
          return SharedBuffer();
        }
        if ((int)v.size() != D) {
          e = nm + " is not [" + std::to_string(D) + "]";
          return SharedBuffer();
        }
        std::copy(v.begin(), v.end(), all.begin() + (std::size_t)b * D);
      }
      return to_f16_(_mc, all);
    });
    if (_band_bias.empty()) { return fail(e); }
    _weight_bytes += (std::size_t)Fin * 4 * (1 + D) + (std::size_t)nb * D * 2;
  }

  // ---- the axial transformers ----
  auto load_block = [&](const std::string& p, Block* blk) {
    Attn& a = blk->attn;
    Ff& f = blk->ff;
    if (!vec16(p + "0.norm.gamma", D, &a.norm) ||
        !mat16(p + "0.to_qkv.weight", 3 * inner, D, &a.qkv) ||
        !mat16(p + "0.to_gates.weight", H, D, &a.gate_w) ||
        !vec16(p + "0.to_gates.bias", H, &a.gate_b) ||
        !mat16(p + "0.to_out.0.weight", D, inner, &a.out) ||
        !vec16(p + "1.net.0.gamma", D, &f.norm) ||
        !mat16(p + "1.net.1.weight", c.ff_inner(), D, &f.up.w) ||
        !vec16(p + "1.net.1.bias", c.ff_inner(), &f.up.b) ||
        !mat16(p + "1.net.4.weight", D, c.ff_inner(), &f.down.w) ||
        !vec16(p + "1.net.4.bias", D, &f.down.b)) {
      return false;
    }
    // The rotary table rides in the checkpoint (a buffer of
    // rotary-embedding-torch's): theta^(-2i/dim_head), kept f32.
    std::string e;
    const std::string fn = p + "0.rotary_embed.freqs";
    a.freqs = ws.derived(std::string(kKey) + "f32|" + fn, [&]() {
      std::vector<float> v;
      if (ws.has(fn)) {
        if (!read_f32_(ws, _mc, fn, &v, nullptr, &e)) {
          return SharedBuffer();
        }
      } else {
        // Absent (a checkpoint saved without its buffers): the library's
        // own table, theta 10000.
        v.resize((std::size_t)Dh / 2);
        for (int i = 0; i < Dh / 2; ++i) {
          v[(std::size_t)i] = (float)(1.0 / std::pow(10000.0,
                                                      (2.0 * i) / Dh));
        }
      }
      if ((int)v.size() != Dh / 2) {
        e = fn + " is not [" + std::to_string(Dh / 2) + "]";
        return SharedBuffer();
      }
      return to_f32_(_mc, v);
    });
    if (a.freqs.empty()) { return fail(e); }
    return true;
  };
  _layers.resize((std::size_t)c.depth);
  for (int l = 0; l < c.depth; ++l) {
    Layer& L = _layers[(std::size_t)l];
    L.time.resize((std::size_t)c.time_depth);
    L.freq.resize((std::size_t)c.freq_depth);
    for (int j = 0; j < c.time_depth; ++j) {
      const std::string p = "layers." + std::to_string(l) + ".0.layers." +
                            std::to_string(j) + ".";
      if (!load_block(p, &L.time[(std::size_t)j])) { return false; }
    }
    for (int j = 0; j < c.freq_depth; ++j) {
      const std::string p = "layers." + std::to_string(l) + ".1.layers." +
                            std::to_string(j) + ".";
      if (!load_block(p, &L.freq[(std::size_t)j])) { return false; }
    }
  }
  if (ws.has("layers." + std::to_string(c.depth) + ".0.layers.0.0.norm."
             "gamma")) {
    return fail("the checkpoint has more layers than the config's depth " +
                std::to_string(c.depth));
  }
  if (!vec16("final_norm.gamma", D, &_final_norm)) { return false; }

  // ---- the mask heads: per Linear, every band's stacked ----
  const int hidden = c.mask_hidden();
  _heads.resize((std::size_t)c.num_stems);
  for (int s = 0; s < c.num_stems; ++s) {
    MaskHead& mh = _heads[(std::size_t)s];
    mh.layers.resize((std::size_t)c.mask_depth);
    for (int li = 0; li < c.mask_depth; ++li) {
      const bool last = li == c.mask_depth - 1;
      const int K = li == 0 ? D : hidden;
      std::string e;
      const std::string base = "mask_estimators." + std::to_string(s) +
                               ".to_freqs.";
      const std::string tail = ".0." + std::to_string(2 * li);
      const std::string key = std::string(kKey) + "mask|" +
                              std::to_string(s) + "|" + std::to_string(li);
      Linear& lin = mh.layers[(std::size_t)li];
      // Rows of band b in the stacked matrix: `hidden` each for a hidden
      // layer; 2 * len_b (its GLU's value and gate halves) for the last,
      // starting at 2 * off_b.
      auto rows_of = [&](int b) {
        return last ? 2 * _band_len[(std::size_t)b] : hidden;
      };
      auto row0_of = [&](int b) {
        return last ? 2 * _band_off[(std::size_t)b] : b * hidden;
      };
      const int total = last ? 2 * Fin : nb * hidden;
      lin.w = ws.derived(key + "|w", [&]() {
        std::vector<float> all((std::size_t)total * K);
        for (int b = 0; b < nb; ++b) {
          const std::string nm = base + std::to_string(b) + tail + ".weight";
          std::vector<float> v;
          std::vector<std::int64_t> sh;
          if (!read_f32_(ws, _mc, nm, &v, &sh, &e)) { return SharedBuffer(); }
          if (!shape_is_(sh, {rows_of(b), K})) {
            e = nm + " is not [" + std::to_string(rows_of(b)) + ", " +
                std::to_string(K) + "] (mask_estimator_depth or "
                "mlp_expansion_factor disagrees with the checkpoint)";
            return SharedBuffer();
          }
          std::copy(v.begin(), v.end(),
                    all.begin() + (std::size_t)row0_of(b) * K);
        }
        return to_f16_(_mc, all);
      });
      if (lin.w.empty()) { return fail(e); }
      lin.b = ws.derived(key + "|b", [&]() {
        std::vector<float> all((std::size_t)total);
        for (int b = 0; b < nb; ++b) {
          const std::string nm = base + std::to_string(b) + tail + ".bias";
          std::vector<float> v;
          if (!read_f32_(ws, _mc, nm, &v, nullptr, &e)) {
            return SharedBuffer();
          }
          if ((int)v.size() != rows_of(b)) {
            e = nm + " does not match its weight";
            return SharedBuffer();
          }
          std::copy(v.begin(), v.end(), all.begin() + row0_of(b));
        }
        return to_f16_(_mc, all);
      });
      if (lin.b.empty()) { return fail(e); }
      _weight_bytes += (std::size_t)total * (K + 1) * 2;
    }
    if (ws.has("mask_estimators." + std::to_string(s) + ".to_freqs.0.0." +
               std::to_string(2 * c.mask_depth) + ".weight")) {
      return fail("the mask head has more layers than the config's "
                  "mask_estimator_depth " + std::to_string(c.mask_depth));
    }
  }
  if (ws.has("mask_estimators." + std::to_string(c.num_stems) +
             ".to_freqs.0.0.0.weight")) {
    return fail("the checkpoint has more mask heads than num_stems " +
                std::to_string(c.num_stems));
  }
  return true;
}

std::size_t
MetalBsRoformer::scratch_bytes(const Config& c)
{
  using U = std::size_t;
  const U T = (U)c.frames(), nb = (U)c.bands(), D = (U)c.dim;
  const U N = T * nb, Fin = (U)c.frame_features();
  const U inner = (U)c.inner(), hidden = (U)c.mask_hidden();
  const U big = N * std::max({3 * inner, (U)c.ff_inner(), hidden});
  U b = T * Fin * 4;                                   // spectrum
  b += 3 * N * D * 2;                                  // x, nx, tmp
  b += big * 2;                                        // qkv / hidden
  if (c.mask_depth > 2) { b += N * hidden * 2; }
  b += N * std::max(inner, D) * 2;                     // att / xt
  b += T * 2 * Fin * 2;                                // mask pre-GLU
  b += (U)c.num_stems * T * Fin * 4;                   // masks
  if (c.skip_connection) { b += N * D * 2; }
  b += T * Fin * 4;                                    // host spectrum
  return b;
}

std::uint64_t
MetalBsRoformer::forward_flops(const Config& c)
{
  const long long T = c.frames(), nb = c.bands(), D = c.dim;
  const long long N = T * nb, Fin = c.frame_features();
  const long long inner = c.inner(), ff = c.ff_inner();
  const long long hidden = c.mask_hidden();
  std::uint64_t f = gemm_flops(T, D, Fin);             // the band split
  const std::uint64_t blk =
      gemm_flops(N, 3 * inner, D) + gemm_flops(N, D, inner) +
      gemm_flops(N, ff, D) + gemm_flops(N, D, ff);
  const std::uint64_t at_time =
      attention_flops(c.heads, nb * T * T, c.dim_head);
  const std::uint64_t at_freq =
      attention_flops(c.heads, T * nb * nb, c.dim_head);
  f += (std::uint64_t)c.depth *
       ((std::uint64_t)c.time_depth * (blk + at_time) +
        (std::uint64_t)c.freq_depth * (blk + at_freq));
  for (int s = 0; s < c.num_stems; ++s) {
    for (int li = 0; li < c.mask_depth; ++li) {
      const bool last = li == c.mask_depth - 1;
      const long long K = li == 0 ? D : hidden;
      f += last ? gemm_flops(T, 2 * Fin, K) : nb * gemm_flops(T, hidden, K);
    }
  }
  return f;
}

bool
MetalBsRoformer::alloc_scratch_(std::string* err)
{
  const Config& c = _cfg;
  using U = std::size_t;
  const U T = (U)c.frames(), nb = (U)c.bands(), D = (U)c.dim;
  const U N = T * nb, Fin = (U)c.frame_features();
  const U inner = (U)c.inner(), hidden = (U)c.mask_hidden();
  const U big = N * std::max({3 * inner, (U)c.ff_inner(), hidden});
  auto mk = [&](U bytes) { return _mc->make_shared_buffer(bytes); };
  _spec = mk(T * Fin * 4);
  _x = mk(N * D * 2);
  _nx = mk(N * D * 2);
  _tmp = mk(N * D * 2);
  _big = mk(big * 2);
  if (c.mask_depth > 2) { _big2 = mk(N * hidden * 2); }
  _att = mk(N * std::max(inner, D) * 2);
  _mo = mk(T * 2 * Fin * 2);
  _mask = mk((U)c.num_stems * T * Fin * 4);
  if (c.skip_connection) { _skip = mk(N * D * 2); }
  _ap_time = mk(sizeof(SteelAttnParams));
  _ap_freq = mk(sizeof(SteelAttnParams));
  if (_spec.empty() || _x.empty() || _nx.empty() || _tmp.empty() ||
      _big.empty() || _att.empty() || _mo.empty() || _mask.empty() ||
      (c.mask_depth > 2 && _big2.empty()) ||
      (c.skip_connection && _skip.empty()) || _ap_time.empty() ||
      _ap_freq.empty()) {
    if (err != nullptr) { *err = "out of memory for the forward's scratch"; }
    return false;
  }
  _yspec.assign(T * Fin, 0.0f);

  // The rotary tables. Every time transformer shares ONE rotary module
  // in the reference (and every frequency one another), so one table per
  // axis, from the first block's frequencies. The angle is the f32
  // product pos * freq, as rotary-embedding-torch forms it; its cos and
  // sin are taken in double and rounded once.
  auto table = [&](const SharedBuffer& freqs, int L) {
    const int half = c.dim_head / 2;
    SharedBuffer b = _mc->make_shared_buffer((std::size_t)L * half * 8);
    if (b.empty()) { return b; }
    const float* f = (const float*)freqs.contents();
    float* d = (float*)b.contents();
    for (int p = 0; p < L; ++p) {
      for (int i = 0; i < half; ++i) {
        const float ang = (float)p * f[i];
        d[((std::size_t)p * half + i) * 2] = (float)std::cos((double)ang);
        d[((std::size_t)p * half + i) * 2 + 1] = (float)std::sin((double)ang);
      }
    }
    return b;
  };
  const SharedBuffer* ft = nullptr;
  const SharedBuffer* ff = nullptr;
  for (const Layer& L : _layers) {
    if (ft == nullptr && !L.time.empty()) { ft = &L.time[0].attn.freqs; }
    if (ff == nullptr && !L.freq.empty()) { ff = &L.freq[0].attn.freqs; }
  }
  // The tables stand for every block, so every block must agree.
  for (const Layer& L : _layers) {
    for (const auto* v : {&L.time, &L.freq}) {
      const SharedBuffer* ref = v == &L.time ? ft : ff;
      for (const Block& b : *v) {
        if (std::memcmp(b.attn.freqs.contents(), ref->contents(),
                        (std::size_t)c.dim_head / 2 * 4) != 0) {
          if (err != nullptr) {
            *err = "the blocks of one axis carry different rotary "
                   "frequencies; one table per axis cannot serve them";
          }
          return false;
        }
      }
    }
  }
  if (ft != nullptr) { _rope_time = table(*ft, (int)T); }
  if (ff != nullptr) { _rope_freq = table(*ff, (int)nb); }
  if ((ft != nullptr && _rope_time.empty()) ||
      (ff != nullptr && _rope_freq.empty())) {
    if (err != nullptr) { *err = "out of memory for the rotary tables"; }
    return false;
  }

  // The two attentions over the one token-major qkv buffer. A token is
  // (frame t, band f) at row t * nb + f; q, k, v are columns [0, inner),
  // [inner, 2 inner), [2 inner, 3 inner) of it, head h at h * Dh.
  const int bq = _attn_nax ? 64 : 32, bk = _attn_nax ? 32 : 16;
  const std::int64_t row = 3 * (std::int64_t)inner, orow = (std::int64_t)inner;
  auto fill = [&](SharedBuffer& pb, int B, int L, std::int64_t sb,
                  std::int64_t sl, std::int64_t ob, std::int64_t ol) {
    auto* p = static_cast<SteelAttnParams*>(pb.contents());
    std::memset(p, 0, sizeof(*p));
    p->B = B; p->H = c.heads; p->D = c.dim_head; p->qL = L; p->kL = L;
    p->gqa_factor = 1;
    p->scale = 1.0f / std::sqrt((float)c.dim_head);
    p->NQ = (L + bq - 1) / bq; p->NK = (L + bk - 1) / bk;
    p->NQ_aligned = L / bq; p->NK_aligned = L / bk;
    p->qL_rem = L - p->NQ_aligned * bq;
    p->kL_rem = L - p->NK_aligned * bk;
    p->qL_off = 0;
    for (auto* s : {p->Q_strides, p->K_strides, p->V_strides}) {
      s[0] = sb; s[1] = c.dim_head; s[2] = sl;
    }
    p->O_strides[0] = ob; p->O_strides[1] = c.dim_head; p->O_strides[2] = ol;
    metal_compute::FunctionConstants fc;
    fc.set_bool(200, (L % bq) == 0).set_bool(201, (L % bk) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false);
    return _lib_attn.function(_attn_nax ? "attn_steel_nax_h_bd64"
                                        : "attn_steel_h_bd64", fc);
  };
  // Time: one sequence per band, its steps `nb` rows apart.
  _fn_attn_time = fill(_ap_time, (int)nb, (int)T, row, (std::int64_t)nb * row,
                       orow, (std::int64_t)nb * orow);
  // Frequency: one sequence per frame, its bands adjacent.
  _fn_attn_freq = fill(_ap_freq, (int)T, (int)nb, (std::int64_t)nb * row, row,
                       (std::int64_t)nb * orow, orow);
  if (!_fn_attn_time.valid() || !_fn_attn_freq.valid()) {
    if (err != nullptr) { *err = "the steel attention did not specialise"; }
    return false;
  }
  return true;
}

void
MetalBsRoformer::gemm_(ComputeEncoder& enc, const SharedBuffer& x,
                       std::size_t xe, const SharedBuffer& w, std::size_t we,
                       const SharedBuffer* b, std::size_t be,
                       const SharedBuffer& y, std::size_t ye, int M, int N,
                       int K, bool* bias_done)
{
  _flops.gemm(M, N, K);
  if (bias_done != nullptr) { *bias_done = true; }
  if (_use_mma2 && M >= _mma_min_m && N >= 16 && M <= mma_row_band(N, K)) {
    // The transformer's projections are TALL -- frames x bands rows
    // (95k a chunk) against a K of 256-1024 -- so the activation is the
    // big operand and the 128x256 tile, re-reading it half as often,
    // wins: MEASURED 2.4-3.4% a chunk on the M5, interleaved.
    // The tree-wide rule (deep K only) was tuned on prefill shapes, and
    // the mask head's GEMMs (M = frames) keep it.
    const bool wide = M >= 16384 || mma_use_wide_tile(N, K);
    const int BN = wide ? 256 : 128;
    enc.set_function(wide ? _fn_mma_wide : _fn_mma);
    enc.set_buffer(0, x, xe * 2);
    enc.set_buffer(1, w, we * 2);
    enc.set_buffer(2, w, we * 2);         // the bias slot, unused
    enc.set_buffer(3, y, ye * 2);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    enc.set_constant(6, M);
    enc.set_constant(7, 0);
    enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                  (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    if (b != nullptr && bias_done != nullptr) {
      *bias_done = false;                // the caller fuses it
    } else if (b != nullptr) {
      enc.set_function(_fn_bias_add);
      enc.set_buffer(0, y, ye * 2);
      enc.set_buffer(1, *b, be * 2);
      enc.set_constant(2, N);
      const unsigned total = (unsigned)((std::size_t)M * N);
      enc.set_constant(3, total);
      enc.dispatch({total, 1, 1}, {256, 1, 1});
    }
    return;
  }
  // Steel: the 64x64 tile wherever M fills it (the widest the M4 has),
  // the 32x32 one under it. Threadgroup-indexed grids.
  const bool w64 = M >= 128;
  const int bm = w64 ? 64 : 32, bn = w64 ? 64 : 32;
  enc.set_function(w64 ? _fn_gemm_64 : _fn_gemm_t);
  enc.set_buffer(0, x, xe * 2);
  enc.set_buffer(1, w, we * 2);
  if (b != nullptr) {
    enc.set_buffer(2, *b, be * 2);
  } else {
    enc.set_buffer(2, w, we * 2);
  }
  enc.set_buffer(3, y, ye * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, b != nullptr ? 1 : 0);
  enc.dispatch({(unsigned)(((N + bn - 1) / bn) * 32),
                (unsigned)(((M + bm - 1) / bm) * 2), 2}, {32, 2, 2});
}

bool
MetalBsRoformer::forward(const float* in, float* out, std::string* err,
                         Taps* taps)
{
  const Config& c = _cfg;
  const int T = c.frames(), nb = c.bands(), D = c.dim, S = c.channels();
  const int N = T * nb, Fin = c.frame_features(), H = c.heads;
  const int Dh = c.dim_head, inner = c.inner(), ffi = c.ff_inner();
  const int hidden = c.mask_hidden(), C = c.chunk_size;
  auto fail = [&](const std::string& what) {
    if (err != nullptr) { *err = what; }
    return false;
  };

  // ---- the spectrum, packed as the band split reads it: frame t's
  // features ordered (bin, channel, re/im) ----
  auto t0 = std::chrono::steady_clock::now();
  stft::Params sp;
  sp.n_fft = c.n_fft;
  sp.hop = c.hop;
  sp.win_length = c.win_length;
  sp.normalized = c.stft_normalized;
  const stft::Stft st(sp);
  if (st.frames(C) != T) { return fail("the STFT geometry disagrees"); }
  float* spec = (float*)_spec.contents();
  const stft::Layout lay{Fin, (std::ptrdiff_t)S * 2};
  for (int ch = 0; ch < S; ++ch) {
    if (!st.forward(in + (std::size_t)ch * C, C, spec + ch * 2, lay)) {
      return fail("the STFT failed");
    }
  }
  _timing.stft_ms = ms_since_(t0);
  if (taps != nullptr) { taps->spec.assign(spec, spec + (std::size_t)T * Fin); }

  // ---- the GPU ----
  t0 = std::chrono::steady_clock::now();
  _flops.begin();
  // The running sum of the layers' outputs starts empty. The previous
  // forward was waited on, so the host may write it.
  if (c.skip_connection) {
    std::memset(_skip.contents(), 0, (std::size_t)N * D * 2);
  }
  metal_compute::CommandStream stream = _mc->make_command_stream();
  ComputeEncoder enc = stream.begin_compute();
  bool gpu_ok = true;
  std::string gpu_err;
  auto sync = [&]() {
    enc.end();
    _flops.flush(stream);
    std::string e;
    if (!stream.commit().wait_ok(&e)) {
      gpu_ok = false;
      if (gpu_err.empty()) { gpu_err = e; }
    }
    enc = stream.begin_compute();
  };
  // VPIPE_BS_ROFORMER_PROFILE=1: a sync at every category boundary and
  // the time since the last one charged to it. Every mark drains the
  // GPU, so the total runs slower than an unprofiled forward; the
  // SHARES are the point.
  static const bool profile =
      std::getenv("VPIPE_BS_ROFORMER_PROFILE") != nullptr;
  struct Bucket { const char* name; double ms; int marks; };
  std::vector<Bucket> buckets;
  auto pclock = std::chrono::steady_clock::now();
  auto mark = [&](const char* name) {
    if (!profile) { return; }
    sync();
    const double ms = ms_since_(pclock);
    bool found = false;
    for (Bucket& b : buckets) {
      if (std::strcmp(b.name, name) == 0) {
        b.ms += ms;
        ++b.marks;
        found = true;
      }
    }
    if (!found) { buckets.push_back(Bucket{name, ms, 1}); }
    pclock = std::chrono::steady_clock::now();
  };
  auto tap = [&](std::vector<float>* dst, const SharedBuffer& src,
                 std::size_t n) {
    sync();
    const auto* p = (const _Float16*)src.contents();
    dst->resize(n);
    for (std::size_t i = 0; i < n; ++i) { (*dst)[i] = (float)p[i]; }
  };
  auto rms = [&](const SharedBuffer& xin, const SharedBuffer& g,
                 const SharedBuffer& y, int rows) {
    enc.set_function(_fn_rms);
    enc.set_buffer(0, xin);
    enc.set_buffer(1, g);
    enc.set_buffer(2, y);
    enc.set_constant(3, D);
    enc.set_constant(4, kNormEps);
    enc.dispatch({512, (unsigned)rows, 1}, {512, 1, 1});
  };
  auto residual = [&](const SharedBuffer& a, const SharedBuffer& b,
                      int n) {
    enc.set_function(_fn_residual);
    enc.set_buffer(0, a);
    enc.set_buffer(1, b);
    enc.set_buffer(2, a);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };
  // One pre-norm attention + feed-forward block along `time` (true) or
  // the bands.
  auto block = [&](const Block& blk, bool time) {
    const Attn& a = blk.attn;
    rms(_x, a.norm, _nx, N);
    gemm_(enc, _nx, 0, a.qkv, 0, nullptr, 0, _big, 0, N, 3 * inner, D);
    mark("norm+qkv");
    // Rotary positions on q and k: a token's frame for the time
    // attention, its band for the frequency one.
    enc.set_function(_fn_rope);
    enc.set_buffer(0, _big);
    enc.set_buffer(1, time ? _rope_time : _rope_freq);
    enc.set_constant(2, 3 * inner);
    enc.set_constant(3, H);
    enc.set_constant(4, Dh);
    enc.set_constant(5, time ? nb : 1);
    enc.set_constant(6, time ? T : nb);
    enc.set_constant(7, N);
    enc.dispatch({(unsigned)(2 * H), (unsigned)N, 1},
                 {(unsigned)(2 * H), (unsigned)std::max(1, 256 / (2 * H)),
                  1});
    mark("rope");
    const int B = time ? nb : T, L = time ? T : nb;
    const int bq = _attn_nax ? 64 : 32;
    _flops.attention(H, (long long)B * L * L, Dh);
    enc.set_function(time ? _fn_attn_time : _fn_attn_freq);
    enc.set_buffer(0, _big, 0);
    enc.set_buffer(1, _big, (std::size_t)inner * 2);
    enc.set_buffer(2, _big, (std::size_t)2 * inner * 2);
    enc.set_buffer(3, _att, 0);
    enc.set_buffer(4, time ? _ap_time : _ap_freq);
    enc.dispatch({32u * (unsigned)((L + bq - 1) / bq), 4u * (unsigned)H,
                  (unsigned)B}, {32, 4, 1});
    mark(time ? "attn.time" : "attn.freq");
    enc.set_function(_fn_gate);
    enc.set_buffer(0, _att);
    enc.set_buffer(1, _nx);
    enc.set_buffer(2, a.gate_w);
    enc.set_buffer(3, a.gate_b);
    enc.set_constant(4, N);
    enc.set_constant(5, D);
    enc.set_constant(6, H);
    enc.set_constant(7, Dh);
    enc.dispatch({(unsigned)N * 32u, 1, 1}, {256, 1, 1});
    gemm_(enc, _att, 0, a.out, 0, nullptr, 0, _tmp, 0, N, D, inner);
    residual(_x, _tmp, N * D);
    mark("gate+out");

    const Ff& f = blk.ff;
    rms(_x, f.norm, _nx, N);
    bool up_biased = true, down_biased = true;
    gemm_(enc, _nx, 0, f.up.w, 0, &f.up.b, 0, _big, 0, N, ffi, D,
          &up_biased);
    const unsigned up_n = (unsigned)((std::size_t)N * ffi);
    if (up_biased) {
      enc.set_function(_fn_gelu);
      enc.set_buffer(0, _big);
      enc.set_buffer(1, _big);
      enc.set_constant(2, N * ffi);
    } else {
      enc.set_function(_fn_bias_gelu);
      enc.set_buffer(0, _big);
      enc.set_buffer(1, f.up.b);
      enc.set_constant(2, ffi);
      enc.set_constant(3, up_n);
    }
    enc.dispatch({up_n, 1, 1}, {256, 1, 1});
    gemm_(enc, _big, 0, f.down.w, 0, &f.down.b, 0, _tmp, 0, N, D, ffi,
          &down_biased);
    if (down_biased) {
      residual(_x, _tmp, N * D);
    } else {
      enc.set_function(_fn_bias_res);
      enc.set_buffer(0, _x);
      enc.set_buffer(1, _tmp);
      enc.set_buffer(2, f.down.b);
      enc.set_constant(3, D);
      const unsigned tot = (unsigned)((std::size_t)N * D);
      enc.set_constant(4, tot);
      enc.dispatch({tot, 1, 1}, {256, 1, 1});
    }
    mark("ff");
  };

  // The band split.
  _flops.add(gemm_flops(T, D, Fin));
  enc.set_function(_fn_split);
  enc.set_buffer(0, _spec);
  enc.set_buffer(1, _band_tab);
  enc.set_buffer(2, _band_gamma);
  enc.set_buffer(3, _band_wt);
  enc.set_buffer(4, _band_bias);
  enc.set_buffer(5, _x);
  enc.set_constant(6, T);
  enc.set_constant(7, Fin);
  enc.set_constant(8, nb);
  enc.set_constant(9, D);
  enc.dispatch({(unsigned)D, (unsigned)((T + 3) / 4), (unsigned)nb},
               {(unsigned)D, 1, 1});
  mark("band_split");
  if (taps != nullptr) {
    tap(&taps->band_split, _x, (std::size_t)N * D);
  }

  for (int l = 0; l < c.depth; ++l) {
    const Layer& L = _layers[(std::size_t)l];
    // MSST's skip_connection: every earlier layer's output is added
    // back in, kept here as their running sum.
    if (c.skip_connection && l > 0) { residual(_x, _skip, N * D); }
    for (const Block& b : L.time) { block(b, true); }
    if (taps != nullptr && l == 0) {
      tap(&taps->layer0_time, _x, (std::size_t)N * D);
    }
    for (const Block& b : L.freq) { block(b, false); }
    if (taps != nullptr && l == 0) {
      tap(&taps->layer0_freq, _x, (std::size_t)N * D);
    }
    if (c.skip_connection) { residual(_skip, _x, N * D); }
  }
  rms(_x, _final_norm, _nx, N);
  if (taps != nullptr) {
    tap(&taps->final_norm, _nx, (std::size_t)N * D);
  }

  // ---- the mask heads, one band at a time on the band-major grid ----
  enc.set_function(_fn_transpose);           // [T, nb, D] -> [nb, T, D]
  enc.set_buffer(0, _nx);
  enc.set_buffer(1, _att);
  enc.set_constant(2, T);
  enc.set_constant(3, nb);
  enc.set_constant(4, D);
  enc.dispatch({(unsigned)D, (unsigned)nb, (unsigned)T}, {(unsigned)D, 1, 1});
  for (int s = 0; s < c.num_stems; ++s) {
    const MaskHead& mh = _heads[(std::size_t)s];
    for (int b = 0; b < nb; ++b) {
      const std::size_t xin = (std::size_t)b * T * D;
      const int len = _band_len[(std::size_t)b];
      const int off = _band_off[(std::size_t)b];
      // Hidden layers ping-pong between _big and _big2.
      const SharedBuffer* src = &_att;
      std::size_t src_e = xin;
      int K = D;
      for (int li = 0; li + 1 < c.mask_depth; ++li) {
        const Linear& lin = mh.layers[(std::size_t)li];
        const SharedBuffer& dst = (li % 2 == 0) ? _big : _big2;
        const std::size_t dst_e = (std::size_t)b * T * hidden;
        gemm_(enc, *src, src_e, lin.w, (std::size_t)b * hidden * K, nullptr,
              0, dst, dst_e, T, hidden, K);
        enc.set_function(_fn_bias_tanh);
        enc.set_buffer(0, dst, dst_e * 2);
        enc.set_buffer(1, lin.b, (std::size_t)b * hidden * 2);
        enc.set_constant(2, hidden);
        const unsigned total = (unsigned)((std::size_t)T * hidden);
        enc.set_constant(3, total);
        enc.dispatch({total, 1, 1}, {256, 1, 1});
        src = &dst;
        src_e = dst_e;
        K = hidden;
      }
      const Linear& last = mh.layers.back();
      const std::size_t mo_e = (std::size_t)T * 2 * off;
      gemm_(enc, *src, src_e, last.w, (std::size_t)2 * off * K, nullptr, 0,
            _mo, mo_e, T, 2 * len, K);
      enc.set_function(_fn_glu);
      enc.set_buffer(0, _mo, mo_e * 2);
      enc.set_buffer(1, last.b, (std::size_t)2 * off * 2);
      enc.set_buffer(2, _mask, (std::size_t)s * T * Fin * 4);
      enc.set_constant(3, T);
      enc.set_constant(4, len);
      enc.set_constant(5, Fin);
      enc.set_constant(6, off);
      enc.dispatch({(unsigned)len, (unsigned)T, 1}, {64, 1, 1});
    }
  }
  mark("mask");
  enc.end();
  _flops.flush(stream);
  {
    std::string e;
    if (!stream.commit().wait_ok(&e)) {
      gpu_ok = false;
      if (gpu_err.empty()) { gpu_err = e; }
    }
  }
  _timing.gpu_ms = ms_since_(t0);
  if (!gpu_ok) { return fail("the GPU reported: " + gpu_err); }
  if (profile) {
    double tot = 0.0;
    for (const Bucket& b : buckets) { tot += b.ms; }
    std::string line;
    for (const Bucket& b : buckets) {
      char buf[96];
      std::snprintf(buf, sizeof buf, " %s %.1f ms (%.1f%%, %d)", b.name,
                    b.ms, tot > 0 ? 100.0 * b.ms / tot : 0.0, b.marks);
      line += buf;
    }
    std::fprintf(stderr, "[bs-roformer profile] %.1f ms:%s\n", tot,
                 line.c_str());
  }
  const float* mask = (const float*)_mask.contents();
  if (taps != nullptr) {
    taps->mask.assign(mask, mask + (std::size_t)c.num_stems * T * Fin);
  }

  // ---- mask x spectrum, then the inverse, per stem and channel ----
  t0 = std::chrono::steady_clock::now();
  for (int s = 0; s < c.num_stems; ++s) {
    const float* ms = mask + (std::size_t)s * T * Fin;
    for (std::size_t i = 0; i < (std::size_t)T * Fin; i += 2) {
      const float xr = spec[i], xi = spec[i + 1];
      const float mr = ms[i], mi = ms[i + 1];
      _yspec[i] = xr * mr - xi * mi;
      _yspec[i + 1] = xr * mi + xi * mr;
    }
    if (c.zero_dc) {
      // Bin 0 of every frame and channel: features [0, 2 * S).
      for (int t = 0; t < T; ++t) {
        std::fill(_yspec.begin() + (std::size_t)t * Fin,
                  _yspec.begin() + (std::size_t)t * Fin + 2 * S, 0.0f);
      }
    }
    for (int ch = 0; ch < S; ++ch) {
      float* y = out + ((std::size_t)s * S + ch) * C;
      if (!st.inverse(_yspec.data() + ch * 2, lay, T, C, y)) {
        return fail("the inverse STFT failed its overlap check");
      }
    }
  }
  _timing.istft_ms = ms_since_(t0);
  return true;
}

}  // namespace vpipe::genai
