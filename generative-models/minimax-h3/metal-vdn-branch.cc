#include "generative-models/minimax-h3/metal-vdn-branch.h"

#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/mma-tile.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace vpipe {
namespace genai {
namespace minimax_h3 {

using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

namespace {

// Cells per thread in vdn_conv_spatial_f32, mirroring VB_CONV_NX there.
// The kernel verifies it, so the two cannot silently disagree.
constexpr int kConvRun = 8;

bool
fail_(std::string* err, const std::string& what)
{
  if (err != nullptr) { *err = what; }
  return false;
}

std::string
prefix_(int layer)
{
  return "transformer_blocks." + std::to_string(layer) + ".attn.";
}

}  // namespace

SharedBuffer
MetalVdnBranch::f32_(const std::string& name, std::string* err)
{
  // A KEPT TRANSFORM, so derived() and not tensor(): the checkpoint is
  // bf16 and every kernel here is fp32. The key names the dtype because
  // that is what changes the bytes -- two models disagreeing about it
  // must not pick up each other's entry.
  //
  // UNLESS THE BLOCKS ARE STREAMED, and then the cached spelling is the
  // wrong one for a reason that would otherwise be invisible: a
  // derived() entry is owned by the SET, so release_block() dropping
  // this object's handles would free nothing at all and read as a fix
  // that works. stream_derived() caches nothing and is counted as
  // streaming throughput, which is also the honest report -- a
  // re-reading model that books its traffic as a one-time load looks
  // like a model that fits.
  auto build = [&]() -> SharedBuffer {
    const MetalLlamaWeights& src = _ws->src();
    const auto* info = src.info(name);
    if (info == nullptr) { return SharedBuffer{}; }
    std::size_t n = 1;
    for (std::int64_t d : info->shape) { n *= (std::size_t)d; }
    SharedBuffer raw = _ws->read(name, _mc, WeightSet::Residency::Copied);
    if (raw.empty()) { return SharedBuffer{}; }
    SharedBuffer dst = _mc->make_shared_buffer(n * sizeof(float));
    if (dst.empty()) { return SharedBuffer{}; }
    float* o = (float*)dst.contents();
    if (info->dtype == "F32") {
      std::memcpy(o, raw.contents(), n * sizeof(float));
    } else if (info->dtype == "BF16") {
      const std::uint16_t* p = (const std::uint16_t*)raw.contents();
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t bits = (std::uint32_t)p[i] << 16;
        std::memcpy(&o[i], &bits, 4);
      }
    } else if (info->dtype == "F16") {
      const __fp16* p = (const __fp16*)raw.contents();
      for (std::size_t i = 0; i < n; ++i) { o[i] = (float)p[i]; }
    } else {
      return SharedBuffer{};
    }
    return dst;
  };
  SharedBuffer out = _stream ? _ws->stream_derived(build)
                             : _ws->derived("vdn/f32/" + name, build);
  if (out.empty()) { fail_(err, "cannot read '" + name + "' as f32"); }
  return out;
}

SharedBuffer
MetalVdnBranch::bf16_(const std::string& name, std::string* err)
{
  // The checkpoint's OWN bytes, when it is already bf16 -- so this is
  // tensor() and not derived(), and the only transform is the one a
  // checkpoint stored some other way would need. That is the whole
  // reason the matrix-core route costs no extra memory: it replaces the
  // fp32 copy rather than sitting beside it.
  const MetalLlamaWeights& src = _ws->src();
  const auto* info = src.info(name);
  if (info == nullptr) {
    fail_(err, "cannot read '" + name + "' as bf16");
    return SharedBuffer{};
  }
  if (info->dtype == "BF16") {
    SharedBuffer out = _stream
                           ? _ws->stream_tensor(name, _mc,
                                                WeightSet::Residency::Copied)
                           : _ws->tensor(name, _mc,
                                         WeightSet::Residency::Copied);
    if (out.empty()) { fail_(err, "cannot read '" + name + "' as bf16"); }
    return out;
  }
  auto build = [&]() -> SharedBuffer {
    std::size_t n = 1;
    for (std::int64_t d : info->shape) { n *= (std::size_t)d; }
    SharedBuffer raw = _ws->read(name, _mc, WeightSet::Residency::Copied);
    if (raw.empty()) { return SharedBuffer{}; }
    SharedBuffer dst = _mc->make_shared_buffer(n * sizeof(std::uint16_t));
    if (dst.empty()) { return SharedBuffer{}; }
    auto* o = (std::uint16_t*)dst.contents();
    // Round to nearest even, as the kernels' own store does -- a
    // truncation here would be a second, different rounding of the
    // same weight depending on which route read it.
    auto narrow = [](float f) {
      std::uint32_t u;
      std::memcpy(&u, &f, 4);
      if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u) {
        return (std::uint16_t)((u >> 16) | 0x0040u);
      }
      return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    };
    if (info->dtype == "F32") {
      const float* p = (const float*)raw.contents();
      for (std::size_t i = 0; i < n; ++i) { o[i] = narrow(p[i]); }
    } else if (info->dtype == "F16") {
      const __fp16* p = (const __fp16*)raw.contents();
      for (std::size_t i = 0; i < n; ++i) { o[i] = narrow((float)p[i]); }
    } else {
      return SharedBuffer{};
    }
    return dst;
  };
  SharedBuffer out = _stream ? _ws->stream_derived(build)
                             : _ws->derived("vdn/bf16/" + name, build);
  if (out.empty()) { fail_(err, "cannot read '" + name + "' as bf16"); }
  return out;
}

std::size_t
MetalVdnBranch::block_bytes_(const Block& b)
{
  std::size_t n = 0;
  for (const SharedBuffer* p :
       {&b.k_sp, &b.k_tm, &b.v_sp, &b.v_tm, &b.beta_proj, &b.a_down,
        &b.a_up, &b.a_dt, &b.a_log, &b.g_down, &b.g_up_w, &b.g_up_b,
        &b.norm, &b.softmax_gate_w, &b.softmax_gate_b, &b.to_out_linear,
        &b.beta_proj_b, &b.g_down_b, &b.g_up_w_b, &b.softmax_gate_wb}) {
    n += p->byte_size();
  }
  return n;
}

void
MetalVdnBranch::release_block(int layer)
{
  if (layer < 0 || layer >= (int)_blocks.size()) { return; }
  _blocks[(std::size_t)layer] = Block{};
}

std::size_t
MetalVdnBranch::resident_bytes() const
{
  std::size_t n = 0;
  for (const Block& b : _blocks) {
    if (b.ready) { n += block_bytes_(b); }
  }
  return n;
}

std::unique_ptr<MetalVdnBranch>
MetalVdnBranch::load(std::shared_ptr<WeightSet> ws,
                     metal_compute::MetalCompute* mc, const vdn::Config& cfg,
                     const Dims& dims, std::string* err)
{
  if (!ws || mc == nullptr || !mc->valid()) {
    fail_(err, "no weight set or metal-compute");
    return nullptr;
  }
  std::string why;
  if (!cfg.supported(&why)) {
    fail_(err, why);
    return nullptr;
  }
  if (cfg.linear_head_dim != dims.head_dim) {
    // Not a limitation of the maths -- the branch may use a narrower
    // head than the backbone -- but of these kernels, which read the
    // backbone's q/k/v with the branch's head_dim and would silently
    // stride wrong.
    fail_(err, "linear_head_dim " + std::to_string(cfg.linear_head_dim)
               + " != the backbone's head_dim "
               + std::to_string(dims.head_dim)
               + "; this port assumes they match");
    return nullptr;
  }
  auto m = std::unique_ptr<MetalVdnBranch>(new MetalVdnBranch());
  m->_ws = std::move(ws);
  m->_mc = mc;
  m->_cfg = cfg;
  m->_dims = dims;
  m->_blocks.resize((std::size_t)(dims.n_layers > 0 ? dims.n_layers : 0));

  m->_lib_branch = mc->load_library("vdn_branch");
  m->_lib_solve  = mc->load_library("vdn_solve");
  m->_sp      = m->_lib_branch.function("vdn_conv_spatial_f32");
  m->_ta      = m->_lib_branch.function("vdn_temporal_act_f32");
  m->_stats   = m->_lib_branch.function("vdn_frame_stats_f32");
  m->_sym     = m->_lib_branch.function("vdn_symmetrise_f32");
  m->_mean    = m->_lib_branch.function("vdn_frame_mean_f32");
  m->_alpha   = m->_lib_branch.function("vdn_alpha_f32");
  m->_bridge  = m->_lib_branch.function("vdn_alpha_bridge_f32");
  m->_gather  = m->_lib_branch.function("vdn_gather_f32");
  m->_readout = m->_lib_branch.function("vdn_readout_f32");
  m->_step    = m->_lib_branch.function("vdn_state_step_f32");
  m->_scale   = m->_lib_branch.function("vdn_scale_f32");
  m->_sgate   = m->_lib_branch.function("vdn_softmax_gate_apply");
  m->_gemm_act = m->_lib_branch.function("vdn_gemm_act_f32");
  m->_chol    = m->_lib_solve.function("vdn_cholesky_f32");
  m->_trinv   = m->_lib_solve.function("vdn_trinv_f32");
  m->_invtr   = m->_lib_solve.function("vdn_inv_and_transition_f32");
  m->_gemm    = m->_lib_solve.function("vdn_gemm_nn_f32");
  // An UNVALIDATED ComputeFunction is a silent no-op -- the dispatch
  // runs and writes nothing -- so refuse to build rather than return an
  // object that produces zeros.
  const bool all = m->_sp.valid() && m->_ta.valid()
                   && m->_stats.valid() && m->_sym.valid()
                   && m->_mean.valid() && m->_alpha.valid()
                   && m->_bridge.valid() && m->_gather.valid()
                   && m->_readout.valid() && m->_step.valid()
                   && m->_scale.valid() && m->_sgate.valid()
                   && m->_gemm_act.valid()
                   && m->_chol.valid()
                   && m->_trinv.valid() && m->_invtr.valid()
                   && m->_gemm.valid();
  if (!all) {
    fail_(err, "the VDN kernels did not load or did not validate");
    return nullptr;
  }
  m->_fail = mc->make_shared_buffer(sizeof(unsigned));
  if (m->_fail.empty()) {
    fail_(err, "cannot allocate the solve failure flag");
    return nullptr;
  }
  std::memset(m->_fail.contents(), 0, sizeof(unsigned));
  m->_dummy = mc->make_shared_buffer(sizeof(float));
  // A REAL zero bias, sized to the widest output any caller asks for.
  // The old gate kernel read bias[c] unconditionally, and the two
  // callers with no bias (beta, whose beta_proj genuinely has none, and
  // the softmax gate's probe) passed the ONE-FLOAT `_dummy` -- so it
  // read up to 7168 floats past the end of a 4-byte buffer. It survived
  // because a fresh Metal page reads as zero, which is luck and not a
  // contract. The GEMM skips the bias entirely when there is none; this
  // stays as the bound-but-unread operand, since an unbound buffer is
  // not reliably null.
  m->_zeros = mc->make_shared_buffer(
      (std::size_t)dims.heads * (std::size_t)cfg.linear_head_dim
      * sizeof(float));
  if (m->_zeros.empty()) {
    fail_(err, "cannot allocate the zero bias");
    return nullptr;
  }
  std::memset(m->_zeros.contents(), 0, m->_zeros.byte_size());

  // THE MATRIX-CORE ROUTE for the three plain GEMMs. Two conditions,
  // and both are load-time because they decide which PRECISION the
  // weights come in at:
  //
  //   the GPU has matrix cores      -- matmul2d is M5 and newer
  //   the features are bf16         -- the operands a matmul2d takes
  //
  // The second is the reference's own dtype map (Dims::bf16_features,
  // default on) and not a concession: an fp32 contraction misses the
  // matrix units entirely, so a branch running the goldens' wide path
  // keeps the ALU kernel and is bit-for-bit what it was.
  //
  // A partial load is REFUSED rather than mixed: half the weights bf16
  // and half fp32 with a route flag that says one thing is how a wrong
  // operand type reaches a kernel that reads it as the other.
  if (dims.bf16_features && mc->supports_matrix_cores()
      && std::getenv("VPIPE_VDN_NO_MMA") == nullptr) {
    m->_lib_mma = mc->load_library("dense_gemm_mma_bf16");
    m->_mma_gemm = m->_lib_mma.function("dense_gemm_mma_t_n128_f16");
    m->_mma_gemm_deep =
        m->_lib_mma.function("dense_gemm_mma_t_n128x256_f16");
    m->_bias_act = m->_lib_branch.function("vdn_bias_act_f32");
    m->_mma = m->_mma_gemm.valid() && m->_mma_gemm_deep.valid()
              && m->_bias_act.valid();
    // The statistics tile IS head_dim -- one [d, d] output per (frame,
    // head), which is one matmul2d tile when d is 128 and would need a
    // tile grid otherwise. The released checkpoint is 128; a different
    // one keeps the fp32 kernel rather than silently reading the wrong
    // extent, and says so.
    if (m->_mma && cfg.linear_head_dim == 128) {
      m->_lib_vdn_mma = mc->load_library("vdn_mma");
      m->_stats_mma = m->_lib_vdn_mma.function("vdn_frame_stats_mma");
      m->_mma_stats = m->_stats_mma.valid();
      m->_readout_mma = m->_lib_vdn_mma.function("vdn_readout_mma");
      m->_readout_norm = m->_lib_branch.function("vdn_readout_norm_f32");
      m->_mma_readout = m->_readout_mma.valid() && m->_readout_norm.valid();
    }
    // The spatial conv. Its own gate, because it asks for one thing the
    // rest does not: kDwBlock must divide the source's channel group, so
    // that a block cannot straddle two heads. head_dim is the group on
    // the transformer's fused projection and the whole width on a tight
    // one, and 16 divides both -- but the check belongs here rather than
    // in the kernel, where an early return writes nothing and reads as
    // a field of zeros.
    if (m->_mma && (cfg.linear_head_dim % kDwBlock) == 0
        && std::getenv("VPIPE_VDN_NO_DW_MMA") == nullptr) {
      m->_lib_conv_mma = mc->load_library("conv2d_mma_bf16");
      static const char* const kNames[3] = {"conv2d_hw_dw5_b16h4_f16",
                                            "conv2d_hw_dw5_b16h6_f16",
                                            "conv2d_hw_dw5_b16h8_f16"};
      bool all = true;
      for (int i = 0; i < 3; ++i) {
        m->_conv_dw[i] = m->_lib_conv_mma.function(kNames[i]);
        all = all && m->_conv_dw[i].valid();
      }
      m->_dw_weight = m->_lib_branch.function("vdn_dw_weight_blockdiag");
      m->_mma_conv = all && m->_dw_weight.valid();
    }
  }
  return m;
}

bool
MetalVdnBranch::block_ready(int layer) const
{
  return layer >= 0 && layer < (int)_blocks.size()
         && _blocks[(std::size_t)layer].ready;
}

bool
MetalVdnBranch::ensure_block(int layer, std::string* err)
{
  if (layer < 0 || layer >= (int)_blocks.size()) {
    return fail_(err, "layer out of range");
  }
  Block& b = _blocks[(std::size_t)layer];
  if (b.ready) { return true; }
  const std::string p = prefix_(layer);
  const std::string la = p + "linear_attention.";

  if (_cfg.conv_k) {
    b.k_sp = f32_(la + "short_conv.k_sp.weight", err);
    b.k_tm = f32_(la + "short_conv.k_tm.weight", err);
  }
  if (_cfg.conv_v) {
    b.v_sp = f32_(la + "short_conv.v_sp.weight", err);
    b.v_tm = f32_(la + "short_conv.v_tm.weight", err);
  }
  // The three plain GEMMs' weights in whichever precision the route
  // that will contract them takes -- INSTEAD OF, not beside. On the
  // matrix-core route the fp32 copy is never read, and loading it would
  // be a second copy of the same numbers charged to the same block.
  if (_mma) {
    b.beta_proj_b = bf16_(la + "beta_proj.weight", err);
    b.g_down_b    = bf16_(la + "output_gate.down.weight", err);
    b.g_up_w_b    = bf16_(la + "output_gate.up.weight", err);
  } else {
    b.beta_proj = f32_(la + "beta_proj.weight", err);
    b.g_down    = f32_(la + "output_gate.down.weight", err);
    b.g_up_w    = f32_(la + "output_gate.up.weight", err);
  }
  b.a_down    = f32_(la + "alpha.down.weight", err);
  b.a_up      = f32_(la + "alpha.up.weight", err);
  b.a_dt      = f32_(la + "alpha.dt_bias", err);
  b.a_log     = f32_(la + "alpha.A_log", err);
  // The BIAS stays fp32 on both routes: it is added after the
  // contraction, by a kernel that is fp32 either way, and it is 7168
  // floats.
  b.g_up_b    = f32_(la + "output_gate.up.bias", err);
  b.norm      = f32_(la + "norm.weight", err);
  // THE ONE TENSOR THAT STAYS bf16, and by some distance the largest.
  // Everything above is widened because the solve reads it: A is only
  // symmetric to the working precision and bf16 pushes the smallest
  // eigenvalue of I + A under the 1 the maths guarantees. This is a
  // plain projection into a bf16 residual stream, run by the
  // TRANSFORMER's GEMM, and widening it would cost 154 MB a block --
  // 7.7 GB over the stack, against 4.28 GB for the whole checkpoint.
  b.to_out_linear =
      _stream ? _ws->stream_tensor(p + "to_out_linear.weight", _mc,
                                   WeightSet::Residency::Copied)
              : _ws->tensor(p + "to_out_linear.weight", _mc,
                            WeightSet::Residency::Copied);
  if (_cfg.enable_softmax_gate) {
    if (_mma) {
      b.softmax_gate_wb = bf16_(p + "softmax_gate.up.weight", err);
    } else {
      b.softmax_gate_w = f32_(p + "softmax_gate.up.weight", err);
    }
    b.softmax_gate_b = f32_(p + "softmax_gate.up.bias", err);
  }

  const SharedBuffer& beta_w = _mma ? b.beta_proj_b : b.beta_proj;
  const SharedBuffer& gd_w   = _mma ? b.g_down_b : b.g_down;
  const SharedBuffer& gu_w   = _mma ? b.g_up_w_b : b.g_up_w;
  const SharedBuffer& sg_w   = _mma ? b.softmax_gate_wb : b.softmax_gate_w;
  const bool ok = !beta_w.empty() && !b.a_down.empty()
                  && !b.a_up.empty() && !b.a_dt.empty() && !b.a_log.empty()
                  && !gd_w.empty() && !gu_w.empty()
                  && !b.g_up_b.empty() && !b.norm.empty()
                  && !b.to_out_linear.empty()
                  && (!_cfg.conv_k || (!b.k_sp.empty() && !b.k_tm.empty()))
                  && (!_cfg.conv_v || (!b.v_sp.empty() && !b.v_tm.empty()))
                  && (!_cfg.enable_softmax_gate
                      || (!sg_w.empty() && !b.softmax_gate_b.empty()));
  if (!ok) {
    return fail_(err, "block " + std::to_string(layer)
                      + ": the linear branch is incomplete in this "
                        "checkpoint");
  }
  b.ready = true;
  return true;
}

// C[M, N] = act(A[M, K] . B[N, K]^T + bias[N]).
//
// TWO ROUTES, and which one runs was settled at LOAD, because it decides
// the precision the weights came in at. On a matrix-core GPU with bf16
// features the contraction is dense_gemm_mma -- the same kernel every
// DiT in this tree runs its projections on -- with the bias and the
// sigmoid following in vdn_bias_act_f32 over a tensor still in cache.
// Everywhere else it is the fp32 tiled GEMM, unchanged and unmoved.
//
// `raw` is where the mma route lands the bare product. It is the
// caller's because the two callers have different widest shapes: the
// branch contracts a frame TILE into 7168 columns, the softmax gate the
// whole packed sequence into 56.
//
// THE MMA ROUTE NEEDS A TIGHT A. A matmul2d tensor's row stride IS its
// contiguous extent, so lda != K cannot be expressed and must not be
// silently read as if it were -- that is a wrong answer rather than a
// slow one. Every caller here passes lda == K already (the offset
// carries the run, not the stride), so this is a guard on a condition
// that holds, which is the only kind worth writing.
void
MetalVdnBranch::gemm_act_(ComputeEncoder& enc, const SharedBuffer& a,
                          std::size_t aoff, int a_elt, int lda,
                          const SharedBuffer& bmat, const SharedBuffer& bias,
                          int use_bias, SharedBuffer& dst, SharedBuffer& raw,
                          int M, int N, int K, int act, int c_elt)
{
  const bool plain = use_bias == 0 && act == 0 && c_elt != 0;
  const bool mma = _mma && a_elt != 0 && lda == K && !bmat.empty()
                   && (plain || raw.byte_size()
                                    >= (std::size_t)M * (std::size_t)N * 2);
  if (mma) {
    // A bias-free, activation-free GEMM whose destination is already
    // bf16 is the product and nothing else, so it goes straight there
    // and the pass below does not run at all. That is the gate's first
    // GEMM, which is why `_lo` narrows on this route.
    SharedBuffer& land = plain ? dst : raw;
    const bool wide = mma_use_wide_tile(N, K);
    const int RN = wide ? 256 : 128;
    const metal_compute::ComputeFunction& fn =
        wide ? _mma_gemm_deep : _mma_gemm;
    // The row band is the mma tiles' 32-bit addressing limit, the rule
    // shared with every other caller of these kernels: an operand whose
    // base passes 2^31 BYTES silently stops storing.
    const int band = mma_row_band(N, K);
    for (int m0 = 0; m0 < M; m0 += band) {
      const int rows = std::min(band, M - m0);
      enc.set_function(fn);
      enc.set_buffer(0, a, aoff + (std::size_t)m0 * (std::size_t)K * 2);
      enc.set_buffer(1, bmat);
      enc.set_buffer(2, bmat);          // bias slot, bound and unread
      enc.set_buffer(3, land, (std::size_t)m0 * (std::size_t)N * 2);
      enc.set_constant(4, K);
      enc.set_constant(5, N);
      enc.set_constant(6, rows);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                    (unsigned)((rows + 127) / 128), 1}, {256, 1, 1});
    }
    if (plain) { return; }
    enc.set_function(_bias_act);
    enc.set_buffer(0, raw);
    enc.set_buffer(1, use_bias != 0 ? bias : _zeros);
    enc.set_buffer(2, dst);
    enc.set_buffer(3, dst);
    enc.set_constant(4, M);
    enc.set_constant(5, N);
    enc.set_constant(6, use_bias);
    enc.set_constant(7, act);
    enc.set_constant(8, c_elt);
    enc.dispatch({(unsigned)(((std::size_t)M * (std::size_t)N + 255) / 256)
                      * 256, 1, 1},
                 {256, 1, 1});
    return;
  }
  enc.set_function(_gemm_act);
  enc.set_buffer(0, a, aoff);
  enc.set_buffer(1, bmat);
  enc.set_buffer(2, bias);
  enc.set_buffer(3, dst);
  enc.set_constant(4, M);
  enc.set_constant(5, N);
  enc.set_constant(6, K);
  enc.set_constant(7, lda);
  enc.set_constant(8, act);
  enc.set_constant(9, use_bias);
  enc.set_buffer(10, a, aoff);
  enc.set_constant(11, a_elt);
  enc.set_buffer(12, dst);
  enc.set_constant(13, c_elt);
  enc.dispatch({(unsigned)((M + 31) / 32) * 16,
                (unsigned)((N + 31) / 32) * 16, 1}, {16, 16, 1});
}

bool
MetalVdnBranch::encode_softmax_gate(ComputeEncoder& enc, int layer, int rows,
                                    const SharedBuffer& x, std::size_t x_off,
                                    bool x_bf16, int hidden_stride,
                                    SharedBuffer& attn_out,
                                    std::size_t out_off, std::string* err)
{
  if (!block_ready(layer)) {
    return fail_(err, "block " + std::to_string(layer) + " is not loaded");
  }
  const Block& b = _blocks[(std::size_t)layer];
  // The config has none. Asked of whichever spelling the route loaded:
  // on the matrix-core route the fp32 one is deliberately absent, and
  // reading it here would turn every gate into a no-op.
  if ((_mma ? b.softmax_gate_wb : b.softmax_gate_w).empty()) {
    return true;
  }
  const int H = _dims.heads, d = _dims.head_dim;
  const int stride = hidden_stride > 0 ? hidden_stride : _dims.hidden;
  if (rows <= 0) { return true; }

  const std::size_t need = (std::size_t)rows * (std::size_t)H * 4;
  if (_sgate_b.byte_size() < need) {
    _sgate_b = _mc->make_shared_buffer(need);
    if (_sgate_b.empty()) {
      return fail_(err, "cannot allocate the softmax gate");
    }
  }
  // The mma route's landing buffer, sized here rather than in reserve()
  // because THIS gate is over the whole packed sequence -- prompt and
  // soundtrack included -- and reserve() is told the video geometry
  // alone. Grown on the same terms as _sgate_b above, and for the same
  // reason: rows does not move within a forward, so it settles on the
  // first block and no later dispatch sees it change.
  if (_mma && _sgate_raw.byte_size() < need / 2) {
    _sgate_raw = _mc->make_shared_buffer(need / 2);
    if (_sgate_raw.empty()) {
      return fail_(err, "cannot allocate the softmax gate product");
    }
  }
  // Direct and per HEAD -- bottleneck 0 -- which is what makes this the
  // same kernel as the linear branch's low-rank per-channel gate rather
  // than a second one. The two differ in granularity because they are
  // asking different questions; see vdn-window-softmax.h.
  // ONE GEMM: direct and per HEAD, so there is no bottleneck to chain
  // through. Same kernel as the linear branch's own gate -- the two
  // differ in granularity, not in shape.
  const int xelt = x_bf16 ? 1 : 0;
  gemm_act_(enc, x, x_off, xelt, stride,
            _mma ? b.softmax_gate_wb : b.softmax_gate_w, b.softmax_gate_b,
            1, _sgate_b, _sgate_raw, rows, H, stride, 1, 0);

  enc.set_function(_sgate);
  enc.set_buffer(0, attn_out, out_off);
  enc.set_buffer(1, _sgate_b);
  enc.set_constant(2, H);
  enc.set_constant(3, d);
  enc.set_constant(4, rows);
  enc.dispatch({(unsigned)rows * (unsigned)H * (unsigned)d, 1, 1},
               {256, 1, 1});
  return true;
}

const metal_compute::SharedBuffer*
MetalVdnBranch::out_linear(int layer) const
{
  if (!block_ready(layer)) { return nullptr; }
  return &_blocks[(std::size_t)layer].to_out_linear;
}

unsigned
MetalVdnBranch::solve_failures() const
{
  return _fail.empty() ? 0u : *(const unsigned*)_fail.contents();
}

void
MetalVdnBranch::clear_solve_failures()
{
  if (!_fail.empty()) { std::memset(_fail.contents(), 0, sizeof(unsigned)); }
}

bool
MetalVdnBranch::reserve(const Geometry& g, std::string* err)
{
  if (g.frames <= 0 || g.grid_h <= 0 || g.grid_w <= 0) {
    return fail_(err, "empty geometry");
  }
  if (_geom.frames == g.frames && _geom.grid_h == g.grid_h
      && _geom.grid_w == g.grid_w && _geom.text_len == g.text_len
      && _reserved_tile == _dims.frame_tile && !_qf.empty()) {
    return true;
  }
  const int H = _dims.heads, d = _cfg.linear_head_dim;
  const int S = g.tokens_per_frame();
  const int C = H * d;
  // The frames the BRANCH owns: under anchor_frames "both" the two
  // anchors leave the input entirely, so every buffer below is sized to
  // the inner clip and the caller's output keeps their rows at zero.
  const int Fi = _cfg.anchors == vdn::AnchorFrames::kBoth
                     ? (g.frames > 2 ? g.frames - 2 : 0)
                     : g.frames;
  if (Fi <= 0) {
    _geom = g;
    return true;                    // the anchors ARE the clip
  }
  // The per-token half is TILED (see kFrameTile): only a few frames of
  // features are live at once, because they are consumed by the
  // per-frame statistics and by a readout that reads a per-frame state.
  // The banks below are the part that is genuinely global.
  const int tile = std::max(1, std::min(Fi, _dims.frame_tile));
  // The low-rank gate's intermediate: [tile rows, bottleneck]. The only
  // caller with a bottleneck is the output gate, which runs a tile at a
  // time; beta and the softmax gate go direct and never touch this.
  _lo = _mc->make_shared_buffer((std::size_t)tile * S * d * sizeof(float));
  if (_lo.empty()) {
    return fail_(err, "cannot allocate the gate bottleneck");
  }
  // Where the matrix-core route lands the OUTPUT GATE's bare product,
  // which is the widest thing it computes: a frame tile's rows into
  // heads * head_dim columns, bf16. Sized for beta's whole-clip shape
  // too, which is narrow (one column per head) but every frame at once
  // -- 2 MB against 29 at generation geometry, so the max costs nothing.
  if (_mma) {
    const std::size_t need =
        std::max((std::size_t)tile * S * C, (std::size_t)Fi * S * H) * 2;
    if (_mmac.byte_size() < need) {
      _mmac = _mc->make_shared_buffer(need);
      if (_mmac.empty()) {
        return fail_(err, "cannot allocate the matmul landing buffer");
      }
    }
    // And the readout's own, which is a tile of the SAME shape. Its own
    // buffer rather than a second use of the one above: they are live in
    // the same tile iteration, and the 29 MB this costs at generation
    // geometry is not worth a lifetime argument between two dispatches
    // whose order is the encoder's rather than anything written down.
    const std::size_t ro = (std::size_t)tile * S * C * 2;
    if (_roraw.byte_size() < ro) {
      _roraw = _mc->make_shared_buffer(ro);
      if (_roraw.empty()) {
        return fail_(err, "cannot allocate the readout product");
      }
    }
  }
  // The dest tile height for THIS patch grid: the padded row count is
  // what the sweep says matters, and a tie goes to the larger tile.
  if (_mma_conv) {
    int best = -1, best_rows = 0;
    for (int i = 0; i < 3; ++i) {
      const int hgt = kDwHeights[i];
      const int rows = ((g.grid_h + hgt - 1) / hgt) * hgt;
      if (best < 0 || rows < best_rows || (rows == best_rows
                                           && hgt > kDwHeights[best])) {
        best = i;
        best_rows = rows;
      }
    }
    _dw_pick = best;
  }
  // The block-diagonal spatial weights. TWO, one per convolved tensor,
  // because both are read inside the same tile iteration and a single
  // buffer would make their lifetimes an argument about dispatch order
  // rather than a fact -- for 5.7 MB each.
  if (_mma_conv) {
    const std::size_t dw =
        (std::size_t)C * kDwBlock * (std::size_t)(5 * 5) * 2;
    for (SharedBuffer* b : {&_dw_k, &_dw_v}) {
      if (b->byte_size() < dw) {
        *b = _mc->make_shared_buffer(dw);
        if (b->empty()) {
          return fail_(err, "cannot allocate the block-diagonal conv "
                            "weights");
        }
      }
    }
  }
  const std::size_t ntile = (std::size_t)tile * S * C;
  const std::size_t nhalo =
      (std::size_t)std::min(Fi, tile + 2 * kConvHalo) * S * C;
  const std::size_t mat = (std::size_t)d * d;
  const std::size_t per = (std::size_t)H * mat;
  auto mk = [&](std::size_t n) { return _mc->make_shared_buffer(n * 4); };
  // The narrow ones. `_state` is NOT among them even though the readout
  // reads it narrow: it is aliased with the solve's fp32 `inv` (below),
  // so it has to be allocated wide and is merely WRITTEN narrow.
  const std::size_t fes = _dims.bf16_features ? 2u : 4u;
  auto mkf = [&](std::size_t n) { return _mc->make_shared_buffer(n * fes); };

  _qf = mkf(ntile); _kf = mkf(ntile); _vf = mkf(ntile);
  // Only for the tensors that ARE convolved: the released config takes
  // both, but a config that convolves one would otherwise carry 58 MB
  // of ring nothing writes.
  if (_cfg.conv_k) { _ring_k = mkf(nhalo); }
  if (_cfg.conv_v) { _ring_v = mkf(nhalo); }
  _beta = mkf((std::size_t)Fi * S * H);
  _A = mk((std::size_t)Fi * per); _B = mk((std::size_t)Fi * per);
  _mean_b = mk((std::size_t)Fi * _dims.hidden);
  _alpha_b = mk((std::size_t)Fi * H * d);
  _tr = mk((std::size_t)Fi * per);
  _inj = mk((std::size_t)Fi * per);
  // THE SOLVE'S THREE INTERMEDIATES SHARE THE SCAN'S BANKS. Each is
  // [Fi, H, d, d] -- 367 MB at production geometry, and the banks are
  // 84% of what is left after the per-token half was tiled -- and each
  // is dead before the buffer it borrows is first written:
  //
  //   L    -> prefix    chol(A) -> L -> Li -> (inv, transition), then
  //   Li   -> suffix    injection = B @ inv. Every one of the three is
  //   inv  -> state     finished before the scan writes a prefix row.
  //
  // The scan then overwrites prefix and suffix in frame order and the
  // gather overwrites state, so nothing reads a borrowed buffer after
  // its lender is done with it. Saves three banks -- 1.1 GB.
  _prefix = mk((std::size_t)Fi * per); _suffix = mk((std::size_t)Fi * per);
  _fb = mk((std::size_t)Fi * H * d); _fa = mk((std::size_t)Fi * H * d);
  _state = mk((std::size_t)Fi * per); _gate_b = mkf(ntile);
  _ones = mk((std::size_t)H * d);
  {
    float* o = (float*)_ones.contents();
    for (std::size_t i = 0; i < (std::size_t)H * d; ++i) { o[i] = 1.0f; }
  }
  if (g.text_len > 0 && _cfg.enable_text_state) {
    // The prompt is a few hundred rows against the video's ~100k and is
    // one chunk, so it is not tiled.
    const std::size_t nt = (std::size_t)g.text_len * C;
    _tkf = mkf(nt); _tvf = mkf(nt);
    _tbeta = mkf((std::size_t)g.text_len * H);
    _tA = mk(per); _tB = mk(per); _tL = mk(per); _tLi = mk(per);
    _tInv = mk(per); _tTr = mk(per); _tState = mk(per);
  } else {
    // The scans still need a start, and a zero one is the no-text
    // semantics. Allocated rather than branched around, so the encode
    // has one shape.
    _tState = mk(per);
    std::memset(_tState.contents(), 0, per * 4);
  }
  if (_qf.empty() || _A.empty() || _prefix.empty() || _suffix.empty()
      || _state.empty() || _tState.empty()) {
    return fail_(err, "cannot allocate the branch scratch");
  }
  _geom = g;
  _reserved_tile = _dims.frame_tile;
  _cached_bounds.clear();
  return true;
}

bool
MetalVdnBranch::encode(ComputeEncoder& enc, int layer, const Geometry& g,
                       const std::vector<vdn::Bound>& bounds,
                       const Inputs& in, SharedBuffer& out, std::string* err)
{
  if (!block_ready(layer)) {
    return fail_(err, "block " + std::to_string(layer) + " is not loaded");
  }
  if (!reserve(g, err)) { return false; }
  if (in.x == nullptr || in.q_raw == nullptr || in.k_raw == nullptr
      || in.v_raw == nullptr || out.empty()) {
    return fail_(err, "missing inputs");
  }
  const Block& b = _blocks[(std::size_t)layer];
  const int H = _dims.heads, d = _cfg.linear_head_dim, hidden = _dims.hidden;
  const int S = g.tokens_per_frame(), C = H * d, K = 5;
  const bool skip = _cfg.anchors == vdn::AnchorFrames::kBoth;
  const int Fi = skip ? g.frames - 2 : g.frames;
  if (Fi <= 0) { return true; }        // the anchors ARE the clip

  std::vector<vdn::Bound> use =
      skip ? vdn::rebase_for_anchor_skip(bounds, g.frames) : bounds;
  if ((int)use.size() != Fi) {
    return fail_(err, "bounds do not match the frame count");
  }
  if (_cached_bounds != use) {
    const vdn::GatherIndex idx = vdn::gather_indices(use, Fi);
    auto ib = [&](const std::vector<int>& v) {
      SharedBuffer s = _mc->make_shared_buffer(v.size() * sizeof(int));
      if (!s.empty()) {
        std::memcpy(s.contents(), v.data(), v.size() * sizeof(int));
      }
      return s;
    };
    std::vector<int> hb(idx.has_before.begin(), idx.has_before.end());
    std::vector<int> ha(idx.has_after.begin(), idx.has_after.end());
    _bidx = ib(idx.before_idx); _aidx = ib(idx.after_idx);
    _hasb = ib(hb);             _hasa = ib(ha);
    _brb  = ib(idx.bridge_before); _bra = ib(idx.bridge_after);
    _cached_bounds = use;
  }

  // The source layout. Defaulted this is the tight fp32 the reference
  // hands over; the transformer passes its fused per-head-grouped bf16
  // projection and its bf16 hidden instead, and nothing below is a
  // special case for either -- the tight shape is these strides' default
  // value. See Inputs.
  const int qes  = in.qkv_bf16 ? 2 : 4;
  const int qrs  = in.qkv_row_stride  > 0 ? in.qkv_row_stride  : C;
  const int qhs  = in.qkv_head_stride > 0 ? in.qkv_head_stride : d;
  const int qelt = in.qkv_bf16 ? 1 : 0;
  const int xelt = in.x_bf16 ? 1 : 0;
  const int oelt = in.out_bf16 ? 1 : 0;
  // The per-token tensors this branch OWNS -- features, the conv
  // rings, beta,
  // the gate, and the state on its way into the readout. See
  // Dims::bf16_features.
  const int felt = _dims.bf16_features ? 1 : 0;
  const std::size_t qrowb = (std::size_t)qrs * (std::size_t)qes;
  const std::size_t xrowb = (std::size_t)hidden * (in.x_bf16 ? 2 : 4);
  const std::size_t anchor_rows = (std::size_t)(skip ? S : 0);
  // Stage timing. Inert unless a caller handed us its stream; see
  // set_profile_stream(). `enc` is the caller's, so a split has to hand
  // it a fresh encoder on the same stream.
  auto pmark = std::chrono::steady_clock::now();
  auto psplit = [&](double& acc) {
    if (_prof_stream == nullptr) { return; }
    enc.end();
    _prof_stream->commit().wait();
    acc += std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - pmark).count();
    enc = _prof_stream->begin_compute();
    pmark = std::chrono::steady_clock::now();
  };

  const std::size_t off_x = in.x_off + anchor_rows * xrowb;
  const std::size_t off_qq = in.q_off + anchor_rows * qrowb;
  const std::size_t off_kk = in.k_off + anchor_rows * qrowb;
  const std::size_t off_vv = in.v_off + anchor_rows * qrowb;
  const unsigned tiles = (unsigned)((d + 31) / 32);
  const int zero = 0, one = 1, outH = H, outC = C, HD = H * d;
  const int tokens = Fi * S;
  const float eps = 1e-6f, half = 0.5f;
  const int per_i = (int)((std::size_t)H * d * d);

  // One tile's features. `in_frames` is the HALOED input range starting
  // at global frame `frame_base`; `out_base` is where the wanted frames
  // begin inside it, and `out_frames` how many. The clip's real length
  // goes in too, so the conv's zero padding is decided by the CLIP's
  // ends and not by the tile's.
  // `beta` is bound at the TILE's base and folded in as sqrt(beta) when
  // the matrix-core statistics will read this tensor -- see
  // vdn_temporal_act_f32 for why half on each side. `_dummy` and a zero
  // flag are the fp32 route, where the statistics kernel applies beta
  // itself while it reads.
  auto features = [&](const SharedBuffer& src, std::size_t soff,
                      const SharedBuffer& wsp, const SharedBuffer& wtm,
                      SharedBuffer& dst, bool conv, bool l2, int in_frames,
                      int spf, int gh, int gw, int frame_base,
                      int total_frames, int out_base, int out_frames,
                      const SharedBuffer* beta = nullptr,
                      std::size_t beta_off = 0,
                      const SharedBuffer* dwb = nullptr,
                      SharedBuffer* ring = nullptr, int* have = nullptr,
                      int nslot = 0) {
    // BOTH POINTER SLOTS ARE ALWAYS BOUND, to the same bytes, and the
    // flag alone picks which one is read. An unbound Metal buffer is not
    // reliably null, so a kernel cannot decide this by testing a pointer
    // -- the same trap the gate's bias hit.
    SharedBuffer& tmp = (ring != nullptr) ? *ring : _ring_k;
    const int slots = (ring != nullptr && nslot > 0) ? nslot : 0;
    // THE SPATIAL CONV RUNS ONCE PER FRAME, not once per tile that wants
    // it. Its output is a RING (see reserve()), so the two halo frames a
    // tile computes past its own body are still there when the next tile
    // arrives -- and only the frames NOT already in the ring are
    // dispatched.
    //
    // `have` is the global frame one past the last one computed, so the
    // new range is [max(frame_base, *have), frame_base + in_frames). It
    // is split into at most TWO dispatches, because a run that crosses
    // the ring's seam is contiguous in neither the source nor the
    // destination -- and splitting it there is what lets both conv
    // kernels stay ring-unaware, taking an offset and a frame count
    // exactly as they did.
    const std::size_t frame_b = (std::size_t)spf * (std::size_t)outC
                                * (felt ? 2u : 4u);
    int fnew = frame_base;
    if (have != nullptr && *have > fnew) { fnew = *have; }
    const int fend = frame_base + in_frames;
    // How many FRAMES one matrix-core conv dispatch may carry.
    //
    // Its tensors are dextents<int32_t, 4>, so MPP computes addresses in
    // 32 bits and an operand whose extent passes 2^31 BYTES silently
    // stops addressing -- the same limit shared/mma-tile.h bands the
    // GEMMs on, in the shape a conv takes it. The widest operand is the
    // ACTIVATION, whose innermost extent is the caller's row stride
    // (3 * inner on a fused projection, not the channel count), so the
    // band is off that and not off the destination.
    //
    // The ring already keeps a run under kFrameTile + 2 * kConvHalo, and
    // at generation geometry the limit is 98 frames -- so this never
    // fires today and costs one min(). It is here because `frame_tile`
    // is a caller's knob and the failure it prevents is silent.
    const std::size_t widest =
        (std::size_t)std::max(qrs, outC) * (std::size_t)spf * 2u;
    const int fband = widest > 0
                          ? (int)std::max<std::size_t>(
                                1, (std::size_t)0x7fffffffu / widest)
                          : 1;
    while (conv && fnew < fend) {
      const int slot = slots > 0 ? fnew % slots : fnew - frame_base;
      int run = slots > 0 ? std::min(fend - fnew, slots - slot)
                          : fend - fnew;
      run = std::min(run, fband);
      const std::size_t src_off =
          soff + (std::size_t)(fnew - frame_base) * (std::size_t)spf
                     * qrowb;
      const std::size_t dst_off = (std::size_t)slot * frame_b;
      // The MATRIX-CORE spatial conv: the same depthwise 5x5, run dense
      // over kDwBlock channels at a time against a block-diagonal
      // weight. It reads the caller's tensor where it lies -- the
      // activation's innermost extent is the ROW STRIDE and the channel
      // origin goes through the source's own group map -- so nothing is
      // repacked, and it writes the same tight ring the ALU kernel does.
      if (_mma_conv && _dw_pick >= 0 && dwb != nullptr && !dwb->empty()
          && qelt != 0 && felt != 0) {
        const int nblk = outC / kDwBlock;
        const int th = kDwHeights[_dw_pick];
        enc.set_function(_conv_dw[_dw_pick]);
        enc.set_buffer(0, src, src_off);
        enc.set_buffer(1, *dwb);
        enc.set_buffer(2, tmp, dst_off);
        enc.set_constant(3, gw);
        enc.set_constant(4, gh);
        enc.set_constant(5, qrs);        // the activation's row stride
        enc.set_constant(6, outC);       // the destination is tight
        enc.set_constant(7, nblk);
        enc.set_constant(8, run);
        enc.set_constant(9, d);          // the source's channel group
        enc.set_constant(10, qhs);       // and the stride between groups
        enc.dispatch({(unsigned)(((gw + kDwTileW - 1) / kDwTileW) * 128),
                      (unsigned)((gh + th - 1) / th),
                      (unsigned)(run * nblk)}, {128, 1, 1});
      } else {
        enc.set_function(_sp);
        enc.set_buffer(0, src, src_off);
        enc.set_buffer(1, wsp);
        enc.set_buffer(2, tmp, dst_off);
        enc.set_constant(3, run);
        enc.set_constant(4, gh);
        enc.set_constant(5, gw);
        enc.set_constant(6, outC);
        enc.set_constant(7, K);
        enc.set_buffer(8, src, src_off);
        enc.set_constant(9, qelt);
        enc.set_constant(10, qrs);
        enc.set_constant(11, qhs);
        enc.set_constant(12, d);
        // Cells per thread. MUST match VB_CONV_NX in the kernel, which
        // checks it and falls back to one-cell-per-thread if it does not
        // -- so a mismatch is slow rather than wrong.
        enc.set_constant(13, kConvRun);
        enc.set_buffer(14, tmp, dst_off);
        enc.set_constant(15, felt);
        const int nxg = (gw + kConvRun - 1) / kConvRun;
        enc.dispatch({(unsigned)(run * gh * nxg * C), 1, 1}, {256, 1, 1});
      }
      fnew += run;
    }
    if (conv) { psplit(_prof.conv); }
    if (conv && have != nullptr) { *have = fend; }
    const int use_t = conv ? 1 : 0, l2i = l2 ? 1 : 0;
    // With the conv on, _ta reads the SPATIAL kernel's own tight output
    // rather than the caller's tensor, so the strides revert to their
    // defaults -- and the element type becomes the BRANCH's, not the
    // caller's.
    const int aelt = conv ? felt : qelt;
    const int arst = conv ? outC : qrs;
    const int ahst = conv ? d : qhs;
    enc.set_function(_ta);
    if (conv) { enc.set_buffer(0, tmp); enc.set_buffer(13, tmp); }
    else      { enc.set_buffer(0, src, soff); enc.set_buffer(13, src, soff); }
    enc.set_buffer(1, conv ? wtm : _dummy);
    enc.set_buffer(2, dst);
    enc.set_constant(3, in_frames);
    enc.set_constant(4, spf);
    enc.set_constant(5, outH);
    enc.set_constant(6, d);
    enc.set_constant(7, K);
    enc.set_constant(8, use_t);
    enc.set_constant(9, l2i);
    enc.set_constant(10, frame_base);
    enc.set_constant(11, total_frames);
    enc.set_constant(12, out_base);
    enc.set_constant(14, aelt);
    enc.set_constant(15, arst);
    enc.set_constant(16, ahst);
    enc.set_buffer(17, dst);
    enc.set_constant(18, felt);
    const int use_beta = beta != nullptr ? 1 : 0;
    // BOTH SLOTS ALWAYS BOUND, to the same bytes, as everywhere else
    // here: an unbound Metal buffer is not reliably null.
    enc.set_buffer(19, beta != nullptr ? *beta : _dummy,
                   beta != nullptr ? beta_off : 0);
    enc.set_buffer(20, beta != nullptr ? *beta : _dummy,
                   beta != nullptr ? beta_off : 0);
    enc.set_constant(21, use_beta);
    enc.set_constant(22, felt);
    // Where this tile's first input frame sits in the ring. Zero slots
    // is the flat buffer, which is what the no-conv path reads.
    const int sbase = (conv && slots > 0) ? frame_base % slots : 0;
    enc.set_constant(23, sbase);
    enc.set_constant(24, conv ? slots : 0);
    enc.dispatch({(unsigned)(out_frames * spf) * 128, (unsigned)H, 1},
                 {128, 1, 1});
  };
  // C = act(A . B^T + bias). Two routes; see gemm_act_.
  auto gemm_act = [&](const SharedBuffer& a, std::size_t aoff, int a_elt,
                      int lda, const SharedBuffer& bmat,
                      const SharedBuffer& bias, int use_bias,
                      SharedBuffer& dst, int Mr, int Nc, int Kd, int act,
                      int c_elt) {
    gemm_act_(enc, a, aoff, a_elt, lda, bmat, bias, use_bias, dst, _mmac,
              Mr, Nc, Kd, act, c_elt);
  };
  // sigmoid(up(down(x)) + b) -- TWO GEMMs, not a per-token loop. The
  // low-rank chain is a matrix multiply and the kernel that walked it
  // per token ran at DRAM rate reusing no weight at all; see
  // vdn_gemm_act_f32. `bneck <= 0` is the direct case (beta, and the
  // softmax half's per-head gate), which is one GEMM.
  auto proj_sigmoid = [&](const SharedBuffer& x, std::size_t xoff,
                          const SharedBuffer& down, const SharedBuffer& up,
                          const SharedBuffer& bias, SharedBuffer& dst,
                          int rows, int outn, int bneck) {
    const bool has_bias = !bias.empty() && &bias != &_dummy;
    if (bneck > 0) {
      // `_lo` stays WIDE on the fp32 route: it is the low-rank chain's
      // own intermediate, consumed immediately by the second GEMM and
      // never stored, so narrowing it would cost accuracy for no
      // traffic.
      //
      // On the matrix-core route it NARROWS, because a matmul2d's
      // operands are 16-bit and the second GEMM contracts this. That is
      // a real extra rounding of a rank-128 intermediate and the reason
      // the two routes are A/B-able rather than assumed equal --
      // measured at the whole branch's output, which is where it lands.
      const int lo_elt = _mma ? 1 : 0;
      gemm_act(x, xoff, xelt, hidden, down, _zeros, 0, _lo, rows, bneck,
               hidden, 0, lo_elt);
      gemm_act(_lo, 0, lo_elt, bneck, up, bias, has_bias ? 1 : 0, dst, rows,
               outn, bneck, 1, felt);
    } else {
      gemm_act(x, xoff, xelt, hidden, up, bias, has_bias ? 1 : 0, dst,
               rows, outn, hidden, 1, felt);
    }
  };
  // A = sum_s beta k k^T and B = sum_s beta v k^T, per (frame, head),
  // followed by the symmetrise the Cholesky downstream depends on.
  //
  // TWO ROUTES. The matrix-core one contracts features that already
  // carry sqrt(beta) and reads one head's slab as a STRIDED view -- see
  // vdn_mma.metal, and note that its `bb` argument goes unread for
  // exactly that reason. The symmetrise still runs on both: A is
  // exactly symmetric out of a matmul2d whose two operands are the same
  // tensor, and that is a property of the matmul rather than one this
  // code should assume.
  auto stats = [&](const SharedBuffer& kk, const SharedBuffer& vv,
                   const SharedBuffer& bb, std::size_t boff,
                   SharedBuffer& aa, std::size_t aoff, SharedBuffer& bo,
                   std::size_t ooff, int spf, int batch) {
    if (_mma_stats) {
      enc.set_function(_stats_mma);
      enc.set_buffer(0, kk); enc.set_buffer(1, vv);
      enc.set_buffer(2, aa, aoff); enc.set_buffer(3, bo, ooff);
      enc.set_constant(4, spf);
      enc.set_constant(5, outH);
      enc.set_constant(6, d);
      enc.dispatch({256, 1, (unsigned)batch}, {256, 1, 1});
    } else {
      enc.set_function(_stats);
      enc.set_buffer(0, kk); enc.set_buffer(1, vv);
      enc.set_buffer(2, bb, boff);
      enc.set_buffer(3, aa, aoff); enc.set_buffer(4, bo, ooff);
      enc.set_constant(5, spf);
      enc.set_constant(6, outH);
      enc.set_constant(7, d);
      enc.set_buffer(8, kk); enc.set_buffer(9, vv);
      enc.set_buffer(10, bb, boff);
      enc.set_constant(11, felt);
      enc.dispatch({tiles * 16, tiles * 16, (unsigned)batch}, {16, 16, 1});
    }
    enc.set_function(_sym);
    enc.set_buffer(0, aa, aoff);
    enc.set_constant(1, d);
    enc.dispatch({(unsigned)((d + 31) / 32) * 32, (unsigned)batch, 1},
                 {32, 1, 1});
  };
  auto solve = [&](const SharedBuffer& aa, const SharedBuffer& al,
                   const SharedBuffer& bo, SharedBuffer& ll, SharedBuffer& lli,
                   SharedBuffer& iv, SharedBuffer& trn, SharedBuffer& ij,
                   int batch) {
    enc.set_function(_chol);
    enc.set_buffer(0, aa); enc.set_buffer(1, ll);
    enc.set_constant(2, d);
    enc.set_buffer(3, _fail);
    enc.dispatch({(unsigned)batch * 128, 1, 1}, {128, 1, 1});
    enc.set_function(_trinv);
    enc.set_buffer(0, ll); enc.set_buffer(1, lli);
    enc.set_constant(2, d);
    enc.dispatch({tiles * 128, (unsigned)batch, 1}, {128, 1, 1});
    enc.set_function(_invtr);
    enc.set_buffer(0, lli); enc.set_buffer(1, al);
    enc.set_buffer(2, iv); enc.set_buffer(3, trn);
    enc.set_constant(4, d);
    enc.dispatch({tiles * 16, tiles * 16, (unsigned)batch}, {16, 16, 1});
    enc.set_function(_gemm);
    enc.set_buffer(0, bo); enc.set_buffer(1, iv); enc.set_buffer(2, ij);
    enc.set_constant(3, d); enc.set_constant(4, d); enc.set_constant(5, d);
    enc.dispatch({tiles * 16, tiles * 16, (unsigned)batch}, {16, 16, 1});
  };

  // beta is per token but only [tokens, heads] wide -- 22 MB at
  // production geometry against 2.89 GB for one feature tensor -- so it
  // is computed whole and the tiling leaves it alone.
  const SharedBuffer& w_beta = _mma ? b.beta_proj_b : b.beta_proj;
  const SharedBuffer& w_gd   = _mma ? b.g_down_b : b.g_down;
  const SharedBuffer& w_gu   = _mma ? b.g_up_w_b : b.g_up_w;
  // The spatial weights expanded block-diagonally, ONCE per forward --
  // they are a property of the layer and the tile loop below reads them
  // every pass. Cheap enough that this is not worth caching per layer:
  // 2.9M stores against the 573 MB the cached form would cost over the
  // stack.
  if (_mma_conv) {
    auto expand = [&](const SharedBuffer& src, SharedBuffer& dst) {
      if (src.empty() || dst.empty()) { return; }
      const int ks = K, blk = kDwBlock;
      enc.set_function(_dw_weight);
      enc.set_buffer(0, src);
      enc.set_buffer(1, dst);
      enc.set_constant(2, outC);
      enc.set_constant(3, ks);
      enc.set_constant(4, blk);
      enc.dispatch({(unsigned)((std::size_t)outC * kDwBlock * ks * ks), 1,
                    1}, {256, 1, 1});
    };
    if (_cfg.conv_k) { expand(b.k_sp, _dw_k); }
    if (_cfg.conv_v) { expand(b.v_sp, _dw_v); }
  }
  proj_sigmoid(*in.x, off_x, _dummy, w_beta, _dummy, _beta, tokens,
               outH, zero);
  psplit(_prof.beta);

  // PASS 1: k/v features and the per-frame statistics, a tile at a time.
  const int tile = std::max(1, std::min(Fi, _dims.frame_tile));
  // The conv rings, and how much of each is already filled. `have` is
  // the global frame ONE PAST the last one convolved, so a tile asks for
  // the frames its halo needs and gets back only the ones its
  // predecessor did not already leave behind.
  const int nslot = std::min(Fi, tile + 2 * kConvHalo);
  int have_k = 0, have_v = 0;
  const std::size_t qframe = (std::size_t)S * qrowb;
  for (int f0 = 0; f0 < Fi; f0 += tile) {
    const int f1 = std::min(f0 + tile, Fi);
    const int h0 = std::max(0, f0 - kConvHalo);
    const int h1 = std::min(Fi, f1 + kConvHalo);
    const std::size_t hk = off_kk + (std::size_t)h0 * qframe;
    const std::size_t hv = off_vv + (std::size_t)h0 * qframe;
    // The statistics kernel indexes its inputs TILE-locally, so beta and
    // the A/B destinations take the offsets instead.
    const std::size_t boff = (std::size_t)f0 * S * H * (felt ? 2u : 4u);
    // On the matrix-core route the features carry sqrt(beta), because a
    // matmul2d has nowhere to put a per-row weight -- so the SCALE is
    // applied by the kernel that writes them and the statistics contract
    // two plain tensors. Everywhere else they are unscaled and the fp32
    // kernel weights them as it reads.
    features(*in.k_raw, hk, b.k_sp, b.k_tm, _kf, _cfg.conv_k, true,
             h1 - h0, S, g.grid_h, g.grid_w, h0, Fi, f0 - h0, f1 - f0,
             _mma_stats ? &_beta : nullptr, boff, &_dw_k, &_ring_k,
             &have_k, nslot);
    features(*in.v_raw, hv, b.v_sp, b.v_tm, _vf, _cfg.conv_v, false,
             h1 - h0, S, g.grid_h, g.grid_w, h0, Fi, f0 - h0, f1 - f0,
             _mma_stats ? &_beta : nullptr, boff, &_dw_v, &_ring_v,
             &have_v, nslot);
    psplit(_prof.features);
    stats(_kf, _vf, _beta, boff, _A, (std::size_t)f0 * per_i * 4, _B,
          (std::size_t)f0 * per_i * 4, S, (f1 - f0) * H);
    psplit(_prof.stats);
  }

  enc.set_function(_mean);
  enc.set_buffer(0, *in.x, off_x);
  enc.set_buffer(1, _mean_b);
  enc.set_constant(2, S);
  enc.set_constant(3, hidden);
  enc.set_buffer(4, *in.x, off_x);
  enc.set_constant(5, xelt);
  enc.dispatch({(unsigned)hidden, (unsigned)Fi, 1}, {64, 1, 1});
  enc.set_function(_alpha);
  enc.set_buffer(0, _mean_b);
  enc.set_buffer(1, b.a_down); enc.set_buffer(2, b.a_up);
  enc.set_buffer(3, b.a_dt);   enc.set_buffer(4, b.a_log);
  enc.set_buffer(5, _alpha_b);
  enc.set_constant(6, hidden);
  enc.set_constant(7, outH);
  enc.set_constant(8, d);
  enc.dispatch({(unsigned)Fi * 128, 1, 1}, {128, 1, 1});
  psplit(_prof.alpha);

  const bool text = _cfg.enable_text_state && g.text_len > 0
                    && in.text_x != nullptr && in.text_k != nullptr
                    && in.text_v != nullptr;
  if (text) {
    // No conv on the prompt: the short conv is a (t, h, w) stencil and
    // text has no grid. Everything else is the video path's, so the
    // delta rule the prompt feeds is the one the frames feed.
    // BETA FIRST on this path, where the video path already had it:
    // the matrix-core route folds sqrt(beta) into the features as they
    // are written, so it has to exist before they are.
    proj_sigmoid(*in.text_x, in.text_x_off, _dummy, w_beta, _dummy,
                 _tbeta, g.text_len, outH, zero);
    features(*in.text_k, in.text_k_off, _dummy, _dummy, _tkf, false, true,
             1, g.text_len, 1, g.text_len, 0, 1, 0, 1,
             _mma_stats ? &_tbeta : nullptr, 0);
    features(*in.text_v, in.text_v_off, _dummy, _dummy, _tvf, false, false,
             1, g.text_len, 1, g.text_len, 0, 1, 0, 1,
             _mma_stats ? &_tbeta : nullptr, 0);
    stats(_tkf, _tvf, _tbeta, 0, _tA, 0, _tB, 0, g.text_len, H);
    solve(_tA, _ones, _tB, _tL, _tLi, _tInv, _tTr, _tState, H);
    enc.set_function(_scale);
    enc.set_buffer(0, _tState);
    enc.set_constant(1, per_i);
    enc.set_constant(2, half);
    enc.dispatch({(unsigned)per_i, 1, 1}, {256, 1, 1});
    psplit(_prof.text);
  }

  // prefix / suffix / state stand in for L / Li / inv -- see reserve().
  solve(_A, _alpha_b, _B, _prefix, _suffix, _state, _tr, _inj, Fi * H);
  psplit(_prof.solve);

  const std::size_t per = (std::size_t)H * d * d;
  auto scan_step = [&](const SharedBuffer& src, std::size_t soff, int f,
                       SharedBuffer& bank) {
    enc.set_function(_step);
    enc.set_buffer(0, src, soff);
    enc.set_buffer(1, _tr, (std::size_t)f * per * 4);
    enc.set_buffer(2, _inj, (std::size_t)f * per * 4);
    enc.set_buffer(3, bank, (std::size_t)f * per * 4);
    enc.set_constant(4, d);
    enc.dispatch({tiles * 16, tiles * 16, (unsigned)H}, {16, 16, 1});
  };
  for (int f = 0; f < Fi; ++f) {
    if (f == 0) { scan_step(_tState, 0, 0, _prefix); }
    else { scan_step(_prefix, (std::size_t)(f - 1) * per * 4, f, _prefix); }
  }
  for (int f = Fi - 1; f >= 0; --f) {
    if (f == Fi - 1) { scan_step(_tState, 0, f, _suffix); }
    else { scan_step(_suffix, (std::size_t)(f + 1) * per * 4, f, _suffix); }
  }
  psplit(_prof.scan);

  enc.set_function(_bridge);
  enc.set_buffer(0, _alpha_b);
  enc.set_buffer(1, _brb); enc.set_buffer(2, _bra);
  enc.set_buffer(3, _fb);  enc.set_buffer(4, _fa);
  enc.set_constant(5, Fi);
  enc.set_constant(6, HD);
  enc.dispatch({(unsigned)HD, 1, 1}, {64, 1, 1});

  const int use_text = text ? 1 : 0;
  enc.set_function(_gather);
  enc.set_buffer(0, _prefix); enc.set_buffer(1, _suffix);
  enc.set_buffer(2, _fb);     enc.set_buffer(3, _fa);
  enc.set_buffer(4, _bidx);   enc.set_buffer(5, _aidx);
  enc.set_buffer(6, _hasb);   enc.set_buffer(7, _hasa);
  enc.set_buffer(8, _tState); enc.set_buffer(9, _state);
  enc.set_constant(10, outH);
  enc.set_constant(11, d);
  enc.set_constant(12, use_text);
  enc.set_buffer(13, _state);
  enc.set_constant(14, felt);
  enc.dispatch({(unsigned)d * 128, (unsigned)H, (unsigned)Fi}, {128, 1, 1});
  psplit(_prof.gather);

  // PASS 2: q's features, the gate and the readout, a tile at a time.
  // q is NOT convolved in the released config, so it needs no halo and
  // is recomputed here rather than carried across the scan -- which is
  // the point: holding it would put a 2.89 GB tensor back.
  for (int f0 = 0; f0 < Fi; f0 += tile) {
    const int f1 = std::min(f0 + tile, Fi);
    const int n = (f1 - f0) * S;
    const std::size_t qoff = off_qq + (std::size_t)f0 * qframe;
    const std::size_t xoff = off_x + (std::size_t)f0 * S * xrowb;
    // `out` is the CALLER's tight fp32 [frames, S, C] -- full frames,
    // anchors included and left at whatever it was zeroed to -- so it
    // keeps the anchor skip in its OWN stride rather than the source's.
    // These two agreed while the input was tight fp32 as well, which is
    // exactly why they had to be separated here.
    const std::size_t ooff = (anchor_rows + (std::size_t)f0 * S)
                             * (std::size_t)C * (in.out_bf16 ? 2 : 4);
    // The gather wrote `_state` at the FEATURE width, so the readout
    // addresses it there too -- the buffer is allocated fp32-wide
    // because it is aliased with the solve's `inv`, but only the low
    // half of each frame's slice is live by now.
    const std::size_t soff2 = (std::size_t)f0 * per_i * (felt ? 2u : 4u);
    features(*in.q_raw, qoff, _dummy, _dummy, _qf, false, true, f1 - f0, S,
             g.grid_h, g.grid_w, f0, Fi, 0, f1 - f0);
    psplit(_prof.features);
    proj_sigmoid(*in.x, xoff, w_gd, w_gu, b.g_up_b, _gate_b, n,
                 outC, d);
    psplit(_prof.gate);
    if (_mma_readout) {
      // The contraction, then the row-wise tail. Split because the
      // RMSNorm needs a whole 128-wide row and a reduction across it,
      // which a cooperative tensor does not hand over -- see
      // vdn_readout_norm_f32. Both are tile-local: `_qf` was written
      // with out_base 0 and `_state` is bound at the tile's base.
      enc.set_function(_readout_mma);
      enc.set_buffer(0, _qf);
      enc.set_buffer(1, _state, soff2);
      enc.set_buffer(2, _roraw);
      enc.set_constant(3, S);
      enc.set_constant(4, outH);
      enc.set_constant(5, d);
      enc.dispatch({256, (unsigned)((S + 127) / 128),
                    (unsigned)((f1 - f0) * H)}, {256, 1, 1});
      enc.set_function(_readout_norm);
      enc.set_buffer(0, _roraw);
      enc.set_buffer(1, _gate_b);
      enc.set_buffer(2, b.norm);
      enc.set_buffer(3, out, ooff);
      enc.set_buffer(4, out, ooff);
      enc.set_constant(5, n);
      enc.set_constant(6, outH);
      enc.set_constant(7, d);
      enc.set_constant(8, eps);
      enc.set_constant(9, oelt);
      enc.set_buffer(10, _gate_b);
      enc.set_constant(11, felt);
      enc.dispatch({(unsigned)n * 128, (unsigned)H, 1}, {128, 1, 1});
    } else {
      enc.set_function(_readout);
      enc.set_buffer(0, _qf);
      enc.set_buffer(1, _state, soff2);
      enc.set_buffer(2, _gate_b);
      enc.set_buffer(3, b.norm);
      enc.set_buffer(4, out, ooff);
      enc.set_constant(5, S);
      enc.set_constant(6, outH);
      enc.set_constant(7, d);
      enc.set_constant(8, eps);
      enc.set_buffer(9, out, ooff);
      enc.set_constant(10, oelt);
      enc.set_buffer(11, _qf);
      enc.set_buffer(12, _state, soff2);
      enc.set_buffer(13, _gate_b);
      enc.set_constant(14, felt);
      // Frame-aligned: z is the frame, x the token tile inside it. `n`
      // is (f1 - f0) * S, so the frame count is what z ranges over.
      enc.dispatch({(unsigned)((S + 7) / 8) * 128, (unsigned)H,
                    (unsigned)(f1 - f0)}, {128, 1, 1});
    }
    psplit(_prof.readout);
  }
  (void)one;
  (void)tokens;
  return true;
}

}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe
