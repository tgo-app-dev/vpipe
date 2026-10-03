#include "generative-models/qwen-image/qwen-image21-trainer.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iterator>

namespace vpipe {
namespace genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

namespace {

// The steel flash-attention parameter block, as attn_steel.metal reads it.
// Mirrored per family, like the model's own.
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

std::uint16_t
bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (std::uint16_t)(u >> 16);
}

float
f32_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

}  // namespace

// The seven projections of a block, in the order modules() lists them.
const QwenImage21Trainer::Proj*
QwenImage21Trainer::projs_()
{
  static const Proj kProjs[kNumProj] = {
      {"attn.to_q", &BlockLora::q, false},
      {"attn.to_k", &BlockLora::k, false},
      {"attn.to_v", &BlockLora::v, false},
      {"attn.to_out.0", &BlockLora::o, false},
      {"img_mlp.gate_layer", &BlockLora::gate, true},
      {"img_mlp.proj", &BlockLora::proj, true},
      {"img_mlp.out", &BlockLora::out, true},
  };
  return kProjs;
}

std::unique_ptr<QwenImage21Trainer>
QwenImage21Trainer::create(MetalQwenImage21Transformer* model,
                           std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return std::unique_ptr<QwenImage21Trainer>();
  };
  if (model == nullptr) { return fail("qwen-image-2.1 trainer: no model"); }
  if (model->_cfg.head_dim != 128) {
    return fail("qwen-image-2.1 trainer: the attention backward is "
                "head_dim 128 only");
  }
  std::unique_ptr<QwenImage21Trainer> t(new QwenImage21Trainer());
  t->_m = model;
  if (const char* e = std::getenv("VPIPE_QWEN_IMAGE21_TRAIN_PROFILE")) {
    t->_prof = *e != '\0' && std::atoi(e) != 0;
  }
  // The A/B for the int8 backward (set_i8_backward); on by default.
  if (const char* e = std::getenv("VPIPE_QWEN_IMAGE21_TRAIN_I8_BWD")) {
    t->_i8_bwd = std::atoi(e) != 0;
  }
  std::string e;
  if (!t->_ops.init(model->_mc, /*bf16=*/true, &e)) { return fail(e); }
  if (model->_quant_bits > 0) {
    t->_lib_deq = model->_mc->load_library("affine_dequant_bf16");
    const std::string g = "g" + std::to_string(model->_quant_group);
    t->_fn_deq4 = t->_lib_deq.function("affine_dequant_w4" + g);
    t->_fn_deq8 = t->_lib_deq.function("affine_dequant_w8" + g);
    if (!t->_fn_deq4.valid() || !t->_fn_deq8.valid()) {
      return fail("qwen-image-2.1 trainer: the dequant kernels for group " +
                  std::to_string(model->_quant_group) + " did not load");
    }
  }
  return t;
}

QwenImage21Trainer::~QwenImage21Trainer()
{
  for (auto& kv : _ab_sets) {
    for (AneBack& a : kv.second) {
      if (a.t != nullptr) { a.t->join(); }
    }
  }
  if (_m != nullptr && _slot >= 0) { _m->release_live_lora(_slot); }
}

std::vector<train::ModuleShape>
QwenImage21Trainer::modules(const std::string& targets) const
{
  const auto& c = _m->_cfg;
  const int H = c.hidden, FF = c.hidden * c.mlp_ratio;
  const bool ff = targets != "attn";
  std::vector<train::ModuleShape> v;
  for (int L = 0; L < c.n_layers; ++L) {
    for (int j = 0; j < kNumProj; ++j) {
      const Proj& p = projs_()[j];
      if (p.ff && !ff) { continue; }
      const std::string leaf = p.leaf;
      int n = H, k = H;
      if (leaf == "img_mlp.gate_layer" || leaf == "img_mlp.proj") {
        n = FF;
      } else if (leaf == "img_mlp.out") {
        k = FF;
      }
      v.push_back({MetalQwenImage21Transformer::block_module(L, leaf), n, k});
    }
  }
  return v;
}

int
QwenImage21Trainer::bind(train::LoraParams* params, std::string* err)
{
  if (params == nullptr) { return 0; }
  const int slot = _m->_lora_slots;
  if (slot >= MetalQwenImage21Transformer::kMaxLoraSlots) {
    if (err != nullptr) {
      *err = "qwen-image-2.1 trainer: both adapter slots are taken by "
             "files; train with at most one bound";
    }
    return 0;
  }
  _p = params;
  const int bound = _m->bind_live_lora(
      slot, [&](const std::string& module, int n, int k, lora::Factors* f) {
        const train::LoraParams::Module* m = params->find(module);
        if (m == nullptr) { return false; }
        if (m->n != n || m->k != k) { return false; }
        f->rank = params->rank();
        f->a = params->a_shadow().subview(m->a_off * 2,
                                          (std::size_t)m->live.rank * k * 2);
        f->b = params->b_shadow().subview(m->b_off * 2,
                                          (std::size_t)n * m->live.rank * 2);
        return true;
      });
  if (bound != (int)params->modules().size()) {
    if (err != nullptr) {
      *err = fmt("qwen-image-2.1 trainer: bound {} of {} modules", bound,
                 params->modules().size())();
    }
    return 0;
  }
  _slot = slot;
  const int NL = _m->_cfg.n_layers;
  _mod_of.assign((std::size_t)NL * kNumProj, nullptr);
  for (int L = 0; L < NL; ++L) {
    for (int j = 0; j < kNumProj; ++j) {
      _mod_of[(std::size_t)L * kNumProj + j] = params->find(
          MetalQwenImage21Transformer::block_module(L, projs_()[j].leaf));
    }
  }
  return bound;
}

std::size_t
QwenImage21Trainer::arena_bytes(const MetalQwenImage21Transformer::Config& c,
                                int rows, int rank, bool keep) noexcept
{
  const std::size_t S = (std::size_t)rows, H = (std::size_t)c.hidden;
  const std::size_t FF = H * (std::size_t)c.mlp_ratio;
  const std::size_t NH = (std::size_t)c.n_heads, NL = (std::size_t)c.n_layers;
  std::size_t b = 0;
  b += NL * S * H * 2;                 // checkpoints
  b += S * H * 2 * 2;                  // final hidden, head scratch
  const std::size_t n = keep ? NL : 1;
  b += n * S * H * 2 * 10;             // kept [S, H] intermediates
  b += n * S * FF * 2 * 3;             // G, U, swiglu
  b += n * NH * S * 4 * 2;             // ml
  b += NH * S * 4 * 2;                 // lse, dsum
  if (!keep) { b += S * H * 2 * 4; }   // the recompute's own scratch
  b += S * H * 4 * 3;                  // dres, dK, dV (f32)
  b += S * H * 2 * 7;                  // da, dh, daf, dO, dQ, dqn, dkn
  b += S * FF * 2 * 3;                 // ds, [dG | dU]
  b += S * H * 2 * 3;                  // stacked dq | dk | dv
  b += H * FF * 2 * 2 * 2;             // W^T (2FF wide) + dequant scratch
  b += S * (std::size_t)std::max(rank, 1) * 2 * 2;   // u, t
  // The attention backward's P^T / dS^T planes (its GEMM form).
  b += train::TrainOps::attn_scratch_bytes((int)NH, (int)S, (int)S);
  return b;
}

bool
QwenImage21Trainer::ensure_arena_(int rows, bool keep)
{
  if (rows <= _rows && keep == _keep_alloc) { return true; }
  metal_compute::MetalCompute* mc = _m->_mc;
  const auto& c = _m->_cfg;
  const std::size_t S = (std::size_t)rows, H = (std::size_t)c.hidden;
  const std::size_t FF = H * (std::size_t)c.mlp_ratio;
  const std::size_t NH = (std::size_t)c.n_heads;
  const std::size_t NL = (std::size_t)c.n_layers;
  const int R = std::max(1, _p != nullptr ? _p->rank() : 1);
  auto bf = [&](std::size_t elems) {
    return mc->make_shared_buffer(elems * 2);
  };
  auto f32 = [&](std::size_t elems) {
    return mc->make_shared_buffer(elems * 4);
  };
  // The kept intermediates: n_layers slices when keeping, one when the
  // backward recomputes each block.
  const std::size_t n = keep ? NL : 1;
  _k.h1 = bf(n * S * H); _k.qpre = bf(n * S * H); _k.kpre = bf(n * S * H);
  _k.qh = bf(n * S * H); _k.kh = bf(n * S * H); _k.vh = bf(n * S * H);
  _k.ao = bf(n * S * H); _k.af = bf(n * S * H);
  _k.x1 = bf(n * S * H); _k.h2 = bf(n * S * H);
  _k.g = bf(n * S * FF); _k.u = bf(n * S * FF); _k.s = bf(n * S * FF);
  _k.mlm = f32(n * NH * S); _k.mls = f32(n * NH * S);
  // Recompute-only scratch (the forward keeps these itself otherwise).
  if (!keep) {
    _vb = bf(S * H); _qn = bf(S * H); _kn = bf(S * H); _tmp = bf(S * H);
  } else {
    _vb = SharedBuffer{}; _qn = SharedBuffer{}; _kn = SharedBuffer{};
    _tmp = SharedBuffer{};
  }
  _ckpt = bf(NL * S * H);
  _final = bf(S * H);
  _modb = bf(2 * 4 * H);
  _nsc = bf(2 * H);
  _xt = bf(S * (std::size_t)c.in_channels);
  _dpred = bf(S * (std::size_t)c.out_channels);
  _part = f32((S * (std::size_t)c.out_channels) / 256 + 2);
  _zero = bf(std::max(4 * H, 2 * FF));
  _tnorm = bf(S * H);
  _dres = f32(S * H);
  _da = bf(S * H); _ds = bf(S * FF); _dgu = bf(S * 2 * FF);
  _dh = bf(S * H); _daf = bf(S * H); _do = bf(S * H); _dqh = bf(S * H);
  _dk32 = f32(S * H); _dv32 = f32(S * H);
  _lse2 = f32(NH * S); _dsum = f32(NH * S);
  _dqn = bf(S * H); _dkn = bf(S * H); _dqkv = bf(S * 3 * H);
  _wt = bf(std::max(2 * FF * H, 3 * H * H));
  _deq = bf(FF * H);
  _lu = bf(S * (std::size_t)R); _lt = bf(S * (std::size_t)R);
  const SharedBuffer* all[] = {
      &_k.h1, &_k.qpre, &_k.kpre, &_k.qh, &_k.kh, &_k.vh, &_k.ao, &_k.af,
      &_k.x1, &_k.h2, &_k.g, &_k.u, &_k.s, &_k.mlm, &_k.mls, &_ckpt,
      &_final, &_modb, &_nsc, &_xt, &_dpred, &_part, &_zero, &_tnorm,
      &_dres, &_da, &_ds, &_dgu, &_dh, &_daf, &_do, &_dqh, &_dk32, &_dv32,
      &_lse2, &_dsum, &_dqn, &_dkn, &_dqkv, &_wt, &_deq, &_lu, &_lt};
  for (const SharedBuffer* b : all) {
    if (b->empty()) { _rows = 0; return false; }
  }
  if (!keep && (_vb.empty() || _qn.empty() || _kn.empty() || _tmp.empty())) {
    _rows = 0;
    return false;
  }
  std::memset(_zero.contents(), 0, _zero.byte_size());
  _rows = rows;
  _keep_alloc = keep;
  return true;
}

const SharedBuffer*
QwenImage21Trainer::dense_(ComputeEncoder& enc, const QWeight& w, int N,
                           int K, QWeight* view)
{
  if (w.is_fp8()) {
    const QWeight* d = _m->fp8_dense_(enc, w, *view, N, K);
    return d != nullptr ? &d->w : nullptr;
  }
  if (!w.quantized) { return &w.w; }
  const metal_compute::ComputeFunction& dq =
      w.bits == 8 ? _fn_deq8 : _fn_deq4;
  if (!dq.valid()) { return nullptr; }
  enc.set_function(dq);
  enc.set_buffer(0, w.codes); enc.set_buffer(1, w.scales);
  enc.set_buffer(2, w.qbias); enc.set_buffer(3, _deq);
  enc.set_constant(4, K); enc.set_constant(5, N);
  const unsigned words = (unsigned)(w.bits == 8 ? (K / 4) : (K / 8));
  enc.dispatch({words, (unsigned)N, 1}, {64, 1, 1});
  return &_deq;
}

bool
QwenImage21Trainer::transpose_to_(ComputeEncoder& enc, const QWeight& w,
                                  int N, int K, int ld, int col,
                                  const SharedBuffer& dst)
{
  QWeight view;
  const SharedBuffer* d = dense_(enc, w, N, K, &view);
  if (d == nullptr) { return false; }
  _ops.transpose(enc, *d, 0, dst, 0, N, K, K, ld, col);
  return true;
}

bool
QwenImage21Trainer::transpose_in_(ComputeEncoder& enc, const QWeight& w,
                                  int N, int K, int ld, int col)
{
  return transpose_to_(enc, w, N, K, ld, col, _wt);
}

bool
QwenImage21Trainer::ane_backward_armed() const noexcept
{
  for (const auto& kv : _ab_sets) {
    for (const AneBack& a : kv.second) {
      if (a.t != nullptr) { return true; }
    }
  }
  return false;
}

int
QwenImage21Trainer::ane_chunk_for_(int M)
{
  int chunk = AneFeedForward::chunk_rows("VPIPE_QWEN_IMAGE21_TRAIN_ANE_CHUNK");
  if (std::getenv("VPIPE_QWEN_IMAGE21_TRAIN_ANE_CHUNK") == nullptr) {
    while (chunk > 256 && chunk * 2 > M) { chunk /= 2; }
  }
  return chunk;
}

bool
QwenImage21Trainer::ane_back_setup_(AneBack& a, int M, int chunk)
{
  if (a.tried) { return a.t != nullptr; }
  a.tried = true;
  metal_compute::MetalCompute* mc = _m->_mc;
  if (mc == nullptr || mc->session() == nullptr) { return false; }
  AneFeedForward::Options o;
  o.session = mc->session();
  o.mc = mc;
  o.tag = a.tag + "-" + std::to_string(chunk);
  o.hidden = a.in;
  o.ffn = a.out;
  o.seq = M;
  o.rows = _m->_cfg.ane_rows;
  o.chunk = chunk;
  o.matmul = true;
  o.profile = std::getenv("VPIPE_QWEN_IMAGE21_ANE_PROFILE") != nullptr;
  a.t = AneFeedForward::create(o);
  return a.t != nullptr;
}

void
QwenImage21Trainer::ane_back_stage_(AneBack& a, int L)
{
  if (a.t == nullptr || a.plan != AneFeedForward::Plan::kSplit ||
      a.wt == nullptr) {
    return;
  }
  AneFfnSource src;
  src.w = a.wt;
  a.t->stage(L, src, 64);
}

bool
QwenImage21Trainer::back_split_(AneBack& a, int L, CommandStream& st,
                                ComputeEncoder& enc, const SharedBuffer& dy,
                                const SharedBuffer& dx, int M,
                                std::string* err)
{
  metal_compute::MetalCompute* mc = _m->_mc;
  auto gpu = [&](int r0, int n) {
    if (n <= 0) { return; }
    QWeight wt;
    wt.w = a.wt->subview(0, (std::size_t)a.out * a.in * 2);
    _m->_i8_hold = !_i8_bwd;
    _m->lin_(enc, _zero, dy, (std::size_t)r0 * a.in, wt, dx,
             (std::size_t)r0 * a.out, n, a.out, a.in);
    _m->_i8_hold = false;
  };
  const AneFeedForward::Plan plan = a.plan;
  if (a.t == nullptr || plan == AneFeedForward::Plan::kGpu) {
    gpu(0, M);
    return true;
  }
  auto drain = [&](double* ms) {
    enc.end();
    const auto t0 = std::chrono::steady_clock::now();
    std::string ge;
    const bool ok = st.commit().wait_ok(&ge);
    if (ms != nullptr) {
      *ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
    }
    st = mc->make_command_stream();
    enc = st.begin_compute();
    if (!ok && err != nullptr) { *err = "qwen-image-2.1 train: " + ge; }
    return ok;
  };
  // dY must be whole before the ANE reads its rows.
  double drain_ms = 0.0;
  if (!drain(&drain_ms)) { return false; }
  _t_drain += drain_ms;
  int arows = 0;
  const auto tj0 = std::chrono::steady_clock::now();
  if (plan == AneFeedForward::Plan::kSplit && a.t->join_stage(L)) {
    _t_join += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - tj0).count();
    arows = a.t->begin(dy, dx, M);
  }
  const auto t_g0 = std::chrono::steady_clock::now();
  gpu(0, M - arows);
  if (!drain(nullptr)) {
    if (arows > 0) { (void)a.t->finish(L, M, 0.0, drain_ms); }
    return false;
  }
  const double t_gpu = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t_g0).count();
  _t_gpu_half += t_gpu;
  const auto tf0 = std::chrono::steady_clock::now();
  if (plan == AneFeedForward::Plan::kProbe) {
    a.t->note_probe(drain_ms, t_gpu);
  }
  // THE GPU FALLBACK for rows the ANE lost: dY is only the tier's input,
  // so every row of it is still there.
  if (!a.t->finish(L, M, t_gpu, drain_ms) && arows > 0) {
    const int lost = a.t->lost_row0();
    if (lost < 0 || lost >= M) {
      if (err != nullptr) {
        *err = "qwen-image-2.1 train: the ANE lost rows it cannot name";
      }
      return false;
    }
    gpu(lost, M - lost);
  }
  _t_finish += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - tf0).count();
  return true;
}

void
QwenImage21Trainer::back_gemm_(ComputeEncoder& enc, const SharedBuffer& dy,
                               int M, int ldw, int K, const SharedBuffer& dx)
{
  // dX [M, K] = dY [M, ldw] (W^T [K, ldw])^T, through the forward's own
  // route: the W^T scratch is an ordinary dense [N = K, K = ldw] weight.
  QWeight wt;
  wt.w = _wt.subview(0, (std::size_t)K * ldw * 2);
  _m->_i8_hold = !_i8_bwd;
  _m->lin_(enc, _zero, dy, 0, wt, dx, 0, M, K, ldw);
  _m->_i8_hold = false;
}

void
QwenImage21Trainer::lora_bwd_(ComputeEncoder& enc, int L,
                              lora::Factors BlockLora::*which,
                              const SharedBuffer& x, std::size_t xe, int ldx,
                              const SharedBuffer& dy, std::size_t dye,
                              int ldy, const SharedBuffer& dx,
                              std::size_t dxe, int lddx, int M, int N, int K)
{
  int j = 0;
  while (j < kNumProj && projs_()[j].lora != which) { ++j; }
  for (int i = 0; i < _m->_lora_slots; ++i) {
    const auto& blk = _m->_lora_blk[i];
    if (L >= (int)blk.size()) { continue; }
    const lora::Factors& f = blk[(std::size_t)L].*which;
    const float s = _m->_lora_scale[i];
    if (f.empty() || s == 0.0f) { continue; }
    const int r = f.rank;
    // u = dY B  [M, r]
    _ops.gemm_nn(enc, dy, dye, f.b, 0, _lu, 0, M, r, N, ldy, r, r, 1.0f,
                 false);
    const train::LoraParams::Module* mod =
        i == _slot ? _mod_of[(std::size_t)L * kNumProj + j] : nullptr;
    if (mod != nullptr) {
      const float c = _p->a_mul();
      // t = x A^T  [M, r], A the shadow (alpha / rank folded in).
      _ops.gemm_nt(enc, x, xe, f.a, 0, _lt, 0, M, r, K, ldx, K, r, 1.0f,
                   false);
      // dB += s dY^T t,  dA += c s u^T x
      const float ls = 1.0f / _loss_scale;
      _ops.gemm_tn_f32(enc, dy, dye, _lt, 0, _p->b_grad(), mod->b_off, M, N,
                       r, ldy, r, r, s * ls, true);
      _ops.gemm_tn_f32(enc, _lu, 0, x, xe, _p->a_grad(), mod->a_off, M, r,
                       K, r, ldx, K, c * s * ls, true);
    }
    // dX += s u A
    _ops.gemm_nn(enc, _lu, 0, f.a, 0, dx, dxe, M, K, r, r, K, lddx, s, true);
  }
}

bool
QwenImage21Trainer::loss(const Sample& smp, double* out, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (smp.layout == nullptr || smp.x0 == nullptr || smp.noise == nullptr ||
      smp.txt == nullptr) {
    return fail("qwen-image-2.1 trainer: an incomplete sample");
  }
  MetalQwenImage21Transformer& m = *_m;
  const qi21::Layout& lay = *smp.layout;
  const int TGT = lay.target_len;
  const int CI = m._cfg.in_channels, CO = m._cfg.out_channels;
  if (CI != CO) {
    return fail("qwen-image-2.1 trainer: the flow target needs in == out "
                "channels");
  }
  if (!ensure_arena_(lay.joint_len, _keep_all)) {
    return fail("qwen-image-2.1 trainer: arena allocation failed");
  }
  const auto* a = static_cast<const std::uint16_t*>(smp.x0->contents());
  const auto* e = static_cast<const std::uint16_t*>(smp.noise->contents());
  {
    auto* d = static_cast<std::uint16_t*>(_xt.contents());
    const float s = smp.sigma;
    for (std::size_t i = 0; i < (std::size_t)TGT * CI; ++i) {
      d[i] = bf16_((1.0f - s) * f32_(a[i]) + s * f32_(e[i]));
    }
  }
  MetalQwenImage21Transformer::Request req;
  req.latents = &_xt;
  req.txt = smp.txt;
  req.layout = &lay;
  req.timestep = smp.sigma;
  std::string ferr;
  SharedBuffer pred = m.forward(req, &ferr);
  if (pred.empty()) {
    if (ferr.empty()) { if (err != nullptr) { err->clear(); } return false; }
    return fail(ferr);
  }
  // The flow target v = noise - x0, as step()'s loss kernel forms it.
  const auto* pp = static_cast<const std::uint16_t*>(pred.contents());
  const std::size_t n = (std::size_t)TGT * CO;
  double sq = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)f32_(pp[i]) - ((double)f32_(e[i]) -
                                            (double)f32_(a[i]));
    sq += d * d;
  }
  *out = smp.weight * sq / (double)n;
  return true;
}

bool
QwenImage21Trainer::step(const Sample& smp, Result* out, std::string* err,
                         bool keep_pred)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (_p == nullptr || _slot < 0) {
    return fail("qwen-image-2.1 trainer: no adapter bound");
  }
  if (smp.layout == nullptr || smp.x0 == nullptr || smp.noise == nullptr ||
      smp.txt == nullptr) {
    return fail("qwen-image-2.1 trainer: an incomplete sample");
  }
  MetalQwenImage21Transformer& m = *_m;
  const qi21::Layout& lay = *smp.layout;
  const auto& c = m._cfg;
  const int ROWS = lay.joint_len;
  const int H = c.hidden, NH = c.n_heads, Hd = c.head_dim;
  const int FF = H * c.mlp_ratio;
  const int NL = c.n_layers;
  const int TGT = lay.target_len;
  const int CI = c.in_channels, CO = c.out_channels;
  const float eps = c.norm_eps;
  if (lay.image_len != TGT) {
    return fail("qwen-image-2.1 trainer: condition-image layouts are not "
                "trained yet; text-to-image only");
  }
  const bool keep = _keep_all;
  if (!ensure_arena_(ROWS, keep)) {
    return fail("qwen-image-2.1 trainer: arena allocation failed");
  }
  metal_compute::MetalCompute* mc = m._mc;
  auto commit = [&](CommandStream& st, ComputeEncoder& enc) {
    enc.end();
    std::string ge;
    if (!st.commit().wait_ok(&ge)) {
      return fail("qwen-image-2.1 train: " + ge);
    }
    return true;
  };

  // ---- x_t, on the host: the sample is small and this is exact -------
  {
    const auto* a = static_cast<const std::uint16_t*>(smp.x0->contents());
    const auto* e = static_cast<const std::uint16_t*>(smp.noise->contents());
    auto* d = static_cast<std::uint16_t*>(_xt.contents());
    const float s = smp.sigma;
    for (std::size_t i = 0; i < (std::size_t)TGT * CI; ++i) {
      d[i] = bf16_((1.0f - s) * f32_(a[i]) + s * f32_(e[i]));
    }
  }

  // ---- forward, keeping the checkpoints ------------------------------
  MetalQwenImage21Transformer::TrainTap tap;
  tap.ckpt = &_ckpt;
  tap.final_hidden = &_final;
  tap.modb = &_modb;
  tap.nsc = &_nsc;
  MetalQwenImage21Transformer::TrainKeep kp;
  if (keep) {
    kp.h1 = &_k.h1; kp.qpre = &_k.qpre; kp.kpre = &_k.kpre;
    kp.qh = &_k.qh; kp.kh = &_k.kh; kp.vh = &_k.vh;
    kp.ao = &_k.ao; kp.af = &_k.af; kp.x1 = &_k.x1; kp.h2 = &_k.h2;
    kp.g = &_k.g; kp.u = &_k.u; kp.s = &_k.s;
    kp.mlm = &_k.mlm; kp.mls = &_k.mls;
    tap.keep = &kp;
  }
  MetalQwenImage21Transformer::Request req;
  req.latents = &_xt;
  req.txt = smp.txt;
  req.layout = &lay;
  req.timestep = smp.sigma;
  req.train = &tap;
  std::string ferr;
  const auto t_f0 = std::chrono::steady_clock::now();
  SharedBuffer pred = m.forward(req, &ferr);
  const auto t_f1 = std::chrono::steady_clock::now();
  if (pred.empty()) {
    if (ferr.empty()) { if (err != nullptr) { err->clear(); } return false; }
    return fail(ferr);
  }
  m.build_rope_(lay, &_rcos, &_rsin);
  if (_rcos.empty() || _rsin.empty()) {
    return fail("qwen-image-2.1 trainer: rope tables failed");
  }

  // Row bands, exactly as the forward modulates them.
  struct Band { int start, rows, mod_row; };
  std::vector<Band> bands;
  if (c.causal_condition && lay.prefix_len > 0 && lay.prefix_len < ROWS) {
    bands.push_back({0, lay.prefix_len, 1});
    bands.push_back({lay.prefix_len, ROWS - lay.prefix_len, 0});
  } else {
    bands.push_back({0, ROWS,
                     (c.causal_condition && lay.prefix_len > 0) ? 1 : 0});
  }
  // The attention's segments, as the forward dispatches them.
  std::vector<train::AttnSegment> segs;
  for (const qi21::Segment& sg : lay.segments) {
    train::AttnSegment a;
    a.q_row0 = sg.start;
    a.qL = sg.end - sg.start;
    a.kL = sg.end;
    a.qL_off = sg.is_text ? sg.start : 0;
    a.causal = sg.is_text;
    segs.push_back(a);
  }
  const float ascale = 1.0f / std::sqrt((float)Hd);

  // ---- loss, and the head's backward ---------------------------------
  const int n_out = TGT * CO;
  // THE LOSS SCALE: a power of two near n / 2 takes dL/dpred to O(1),
  // where the ANE's fp16 represents it. Exact to undo, and a no-op for a
  // GPU-only backward, whose bf16 has f32's exponent range.
  const bool ane_bwd = _ane_bwd && _m->_mc->session() != nullptr;
  _loss_scale = 1.0f;
  if (ane_bwd) {
    _loss_scale = std::exp2(std::round(std::log2(0.5 * (double)n_out)));
  }
  const float coef = 2.0f * smp.weight * _loss_scale / (float)n_out;
  int groups = 0;
  {
    CommandStream st = mc->make_command_stream();
    ComputeEncoder enc = st.begin_compute();
    groups = _ops.flow_loss(enc, pred, *smp.x0, *smp.noise, _dpred, _part,
                            n_out, coef);
    _ops.fill_f32(enc, _dres, 0, 0.0f, (std::size_t)ROWS * H);
    // d(normed) = dpred proj_out: W^T is [H, CO].
    if (!transpose_in_(enc, m._proj_out, CO, H, CO, 0)) {
      return fail("qwen-image-2.1 trainer: proj_out is not readable");
    }
    back_gemm_(enc, _dpred, TGT, CO, H, _tnorm);
    // norm_out's backward, target rows only; they read modulation row 0.
    const std::size_t t0 = (std::size_t)lay.prefix_len * H;
    _ops.ln_mod_bwd(enc, _final, t0, _tnorm, 0, false, _nsc, 0, _dres, t0,
                    TGT, H, eps);
    if (!commit(st, enc)) { return false; }
  }
  double sq = 0.0;
  const auto* pp = static_cast<const float*>(_part.contents());
  for (int i = 0; i < groups; ++i) { sq += (double)pp[i]; }
  const double loss = smp.weight * sq / (double)n_out;

  // This step's tier set, by the chunk its sequence splits in.
  AneBack* ab_set = nullptr;
  if (ane_bwd) {
    const int chunk = ane_chunk_for_(ROWS);
    auto& set = _ab_sets[chunk];
    const int dims[3][2] = {{H, FF}, {2 * FF, H}, {3 * H, H}};
    const char* tags[3] = {"qwen-image-2.1-train-down",
                           "qwen-image-2.1-train-gateup",
                           "qwen-image-2.1-train-qkv"};
    for (int i = 0; i < 3; ++i) {
      AneBack& a = set[(std::size_t)i];
      if (_ab_wt[i].empty()) {
        _ab_wt[i] = mc->make_shared_buffer(
            (std::size_t)dims[i][0] * dims[i][1] * 2);
      }
      if (!a.tried) {
        a.tag = tags[i];
        a.in = dims[i][0];
        a.out = dims[i][1];
      }
      a.wt = _ab_wt[i].empty() ? nullptr : &_ab_wt[i];
      if (a.wt != nullptr) { (void)ane_back_setup_(a, ROWS, chunk); }
    }
    ab_set = set.data();
  }

  // ---- the blocks, in reverse -----------------------------------------
  const bool nax = m._use_attn_nax;
  const int A_BQ = nax ? 64 : 32;
  const int A_BK = nax ? 32 : 16;
  const std::size_t RH = (std::size_t)ROWS * H;

  for (int L = NL - 1; L >= 0; --L) {
    if (_stop && _stop()) {
      if (err != nullptr) { err->clear(); }
      return false;
    }
    const bool held = L < (int)m._blocks.size() &&
                      !m._blocks[(std::size_t)L].qw.empty();
    const Block* b = held ? &m._blocks[(std::size_t)L] : m._slots.acquire(L);
    if (b == nullptr) {
      return fail("qwen-image-2.1 trainer: a streamed block did not load");
    }
    auto LB = [&](lora::Factors BlockLora::*w) { return m.lora_blk_(L, w); };
    const std::size_t x0e = (std::size_t)L * RH;

    CommandStream st = mc->make_command_stream();
    ComputeEncoder enc = st.begin_compute();
    // VPIPE_QWEN_IMAGE21_TRAIN_PROFILE: commit at each category boundary
    // and charge the time to it. It commits ~15 times a block, so a
    // profiled step is not a timing step -- read the shares.
    auto pclock = std::chrono::steady_clock::now();
    auto mark = [&](const char* bucket) {
      if (!_prof) { return; }
      enc.end();
      std::string ge;
      (void)st.commit().wait_ok(&ge);
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - pclock).count();
      bool found = false;
      for (auto& bk : _pbuckets) {
        if (bk.first == bucket) { bk.second += ms; found = true; }
      }
      if (!found) { _pbuckets.emplace_back(bucket, ms); }
      st = mc->make_command_stream();
      enc = st.begin_compute();
      pclock = std::chrono::steady_clock::now();
    };

    // This block's intermediates: its slice of the kept arenas, or the
    // one set the recompute below fills.
    const std::size_t sl = keep ? (std::size_t)L : 0;
    const std::size_t FFe = (std::size_t)FF;
    const std::size_t MLe = (std::size_t)NH * ROWS;
    struct Views {
      SharedBuffer h1, qpre, kpre, qh, kh, vh, ao, af, x1, h2, g, u, s, mlm,
          mls;
    } V;
    V.h1 = _k.h1.subview(sl * RH * 2, RH * 2);
    V.qpre = _k.qpre.subview(sl * RH * 2, RH * 2);
    V.kpre = _k.kpre.subview(sl * RH * 2, RH * 2);
    V.qh = _k.qh.subview(sl * RH * 2, RH * 2);
    V.kh = _k.kh.subview(sl * RH * 2, RH * 2);
    V.vh = _k.vh.subview(sl * RH * 2, RH * 2);
    V.ao = _k.ao.subview(sl * RH * 2, RH * 2);
    V.af = _k.af.subview(sl * RH * 2, RH * 2);
    V.x1 = _k.x1.subview(sl * RH * 2, RH * 2);
    V.h2 = _k.h2.subview(sl * RH * 2, RH * 2);
    V.g = _k.g.subview(sl * ROWS * FFe * 2, ROWS * FFe * 2);
    V.u = _k.u.subview(sl * ROWS * FFe * 2, ROWS * FFe * 2);
    V.s = _k.s.subview(sl * ROWS * FFe * 2, ROWS * FFe * 2);
    V.mlm = _k.mlm.subview(sl * MLe * 4, MLe * 4);
    V.mls = _k.mls.subview(sl * MLe * 4, MLe * 4);

    // ---- recompute, keeping what the backward reads ------------------
    auto ln_mod = [&](const SharedBuffer& x, std::size_t xe,
                      const SharedBuffer& scale, std::size_t se,
                      const SharedBuffer& y, std::size_t ye, int R) {
      enc.set_function(m._fn_ln_mod);
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, scale, se * 2);
      enc.set_buffer(2, _zero); enc.set_buffer(3, y, ye * 2);
      enc.set_constant(4, H); enc.set_constant(5, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    auto rms = [&](const SharedBuffer& x, const SharedBuffer& w,
                   const SharedBuffer& y, int R, int D) {
      enc.set_function(m._fn_rms);
      enc.set_buffer(0, x); enc.set_buffer(1, w); enc.set_buffer(2, y);
      enc.set_constant(3, D); enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    auto transpose = [&](const SharedBuffer& in, const SharedBuffer& o,
                         int A, int B) {
      enc.set_function(m._fn_transpose);
      enc.set_buffer(0, in); enc.set_buffer(1, o);
      enc.set_constant(2, A); enc.set_constant(3, B);
      enc.set_constant(4, Hd);
      enc.dispatch({(unsigned)Hd, (unsigned)B, (unsigned)A},
                   {(unsigned)Hd, 1, 1});
    };
    auto gated = [&](const SharedBuffer& h, std::size_t he,
                     const SharedBuffer& gate, std::size_t ge,
                     const SharedBuffer& sub, std::size_t se, int R) {
      if ((H % 4) == 0 && m._fn_gated_tanh4.valid()) {
        enc.set_function(m._fn_gated_tanh4);
        enc.set_buffer(0, h, he * 2); enc.set_buffer(1, gate, ge * 2);
        enc.set_buffer(2, sub, se * 2);
        enc.set_constant(3, H / 4); enc.set_constant(4, R);
        enc.dispatch({(unsigned)(H / 4), (unsigned)R, 1}, {256, 1, 1});
        return;
      }
      enc.set_function(m._fn_gated_tanh);
      enc.set_buffer(0, h, he * 2); enc.set_buffer(1, gate, ge * 2);
      enc.set_buffer(2, sub, se * 2);
      enc.set_constant(3, H);
      enc.set_constant(4, (int)((std::size_t)R * H));
      enc.dispatch({(unsigned)((std::size_t)R * H), 1, 1}, {256, 1, 1});
    };
    auto tr_rope = [&](const SharedBuffer& in, const SharedBuffer& o) {
      enc.set_function(m._fn_tr_rope);
      enc.set_buffer(0, in); enc.set_buffer(1, o);
      enc.set_buffer(2, _rcos); enc.set_buffer(3, _rsin);
      enc.set_constant(4, NH); enc.set_constant(5, ROWS);
      enc.set_constant(6, Hd);
      enc.dispatch({(unsigned)(Hd / 2), (unsigned)ROWS, (unsigned)NH},
                   {(unsigned)(Hd / 2), 1, 1});
    };

    if (!keep) {
    for (const Band& bd : bands) {
      ln_mod(_ckpt, x0e + (std::size_t)bd.start * H, _modb,
             (std::size_t)bd.mod_row * 4 * H, V.h1, (std::size_t)bd.start * H,
             bd.rows);
    }
    m.lin_(enc, _zero, V.h1, 0, b->qw, V.qpre, 0, ROWS, H, H,
           LB(&BlockLora::q));
    m.lin_(enc, _zero, V.h1, 0, b->kw, V.kpre, 0, ROWS, H, H,
           LB(&BlockLora::k));
    m.lin_(enc, _zero, V.h1, 0, b->vw, _vb, 0, ROWS, H, H, LB(&BlockLora::v));
    mark("re.qkv");
    rms(V.qpre, b->nq, _qn, ROWS * NH, Hd);
    rms(V.kpre, b->nk, _kn, ROWS * NH, Hd);
    tr_rope(_qn, V.qh);
    tr_rope(_kn, V.kh);
    transpose(_vb, V.vh, ROWS, NH);
    for (const train::AttnSegment& sg : segs) {
      SteelAttnParams p{};
      p.B = 1; p.H = NH; p.D = Hd; p.qL = sg.qL; p.kL = sg.kL;
      p.gqa_factor = 1; p.scale = ascale;
      p.NQ = (sg.qL + A_BQ - 1) / A_BQ; p.NK = (sg.kL + A_BK - 1) / A_BK;
      p.NQ_aligned = sg.qL / A_BQ; p.NK_aligned = sg.kL / A_BK;
      p.qL_rem = sg.qL - p.NQ_aligned * A_BQ;
      p.kL_rem = sg.kL - p.NK_aligned * A_BK;
      p.qL_off = sg.qL_off;
      const std::int64_t hs = (std::int64_t)ROWS * Hd;
      p.Q_strides[0] = (std::int64_t)NH * hs; p.Q_strides[1] = hs;
      p.Q_strides[2] = Hd;
      p.K_strides[0] = p.Q_strides[0]; p.K_strides[1] = hs;
      p.K_strides[2] = Hd;
      p.V_strides[0] = p.Q_strides[0]; p.V_strides[1] = hs;
      p.V_strides[2] = Hd;
      p.O_strides[0] = p.Q_strides[0]; p.O_strides[1] = hs;
      p.O_strides[2] = Hd;
      metal_compute::FunctionConstants fc;
      fc.set_bool(200, (sg.qL % A_BQ) == 0).set_bool(201, (sg.kL % A_BK) == 0)
          .set_bool(300, false).set_bool(301, sg.causal)
          .set_bool(302, false).set_bool(304, true);
      metal_compute::ComputeFunction fn =
          nax ? m._lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc)
              : m._lib_attn.function("attn_steel_h_bd128_bf16", fc);
      if (!fn.valid()) {
        return fail("qwen-image-2.1 trainer: the exporting attention kernel "
                    "did not specialize");
      }
      enc.set_function(fn);
      const std::size_t qo = (std::size_t)sg.q_row0 * Hd * 2;
      enc.set_buffer(0, V.qh, qo); enc.set_buffer(1, V.kh);
      enc.set_buffer(2, V.vh); enc.set_buffer(3, V.ao, qo);
      enc.set_constant(4, p);
      const std::size_t mo = (std::size_t)NH * sg.q_row0 * 4;
      enc.set_buffer(12, V.mlm, mo); enc.set_buffer(13, V.mls, mo);
      enc.dispatch({(unsigned)(32 * p.NQ), (unsigned)(4 * NH), 1},
                   {32, 4, 1});
    }
    transpose(V.ao, V.af, NH, ROWS);
    mark("re.attn");
    m.lin_(enc, _zero, V.af, 0, b->ow, _tmp, 0, ROWS, H, H, LB(&BlockLora::o));
    // x1 = x + tanh(g1) attn, in place on a copy of the checkpoint.
    enc.set_function(m._fn_copy_rows);
    enc.set_buffer(0, _ckpt); enc.set_buffer(1, V.x1);
    enc.set_constant(2, (int)x0e); enc.set_constant(3, 0);
    enc.set_constant(4, ROWS); enc.set_constant(5, H);
    enc.set_constant(6, H); enc.set_constant(7, H);
    enc.dispatch({(unsigned)RH, 1, 1}, {256, 1, 1});
    for (const Band& bd : bands) {
      gated(V.x1, (std::size_t)bd.start * H, _modb,
            (std::size_t)bd.mod_row * 4 * H + H, _tmp,
            (std::size_t)bd.start * H, bd.rows);
    }
    for (const Band& bd : bands) {
      ln_mod(V.x1, (std::size_t)bd.start * H, _modb,
             (std::size_t)bd.mod_row * 4 * H + 2 * H, V.h2,
             (std::size_t)bd.start * H, bd.rows);
    }
    m.lin_(enc, _zero, V.h2, 0, b->gate_w, V.g, 0, ROWS, FF, H,
           LB(&BlockLora::gate));
    m.lin_(enc, _zero, V.h2, 0, b->proj_w, V.u, 0, ROWS, FF, H,
           LB(&BlockLora::proj));
    mark("re.o+ff");
    {
      // The forward's own choice of width, so S is the forward's S.
      const std::size_t n = (std::size_t)ROWS * FF;
      const bool v4 = (n % 4) == 0 && m._fn_swiglu4.valid();
      enc.set_function(v4 ? m._fn_swiglu4 : m._fn_swiglu);
      enc.set_buffer(0, V.g); enc.set_buffer(1, V.u); enc.set_buffer(2, V.s);
      enc.set_constant(3, (int)(v4 ? n / 4 : n));
      enc.dispatch({(unsigned)(v4 ? n / 4 : n), 1, 1}, {256, 1, 1});
    }

    }  // the recompute

    // ---- the ANE's plan for this block's three backward GEMMs --------
    //
    // Each tier stages its OWN transposed weight, so all three are
    // written now and must have landed before the worker reads the first.
    // The first stages at once; each later one stages as soon as the tier
    // before it has finished predicting, under the GPU work in between --
    // and the q|k|v one under the whole attention backward.
    const bool ab = ab_set != nullptr &&
                    (ab_set[0].t != nullptr || ab_set[1].t != nullptr ||
                     ab_set[2].t != nullptr);
    AneBack* abt = ab_set;
    if (ab) {
      for (int i = 0; i < 3; ++i) {
        AneBack& a = abt[i];
        a.plan = a.t != nullptr ? a.t->plan_block()
                                : AneFeedForward::Plan::kGpu;
      }
      if (!transpose_to_(enc, b->out_w, H, FF, H, 0, _ab_wt[0]) ||
          !transpose_to_(enc, b->gate_w, FF, H, 2 * FF, 0, _ab_wt[1]) ||
          !transpose_to_(enc, b->proj_w, FF, H, 2 * FF, FF, _ab_wt[1]) ||
          !transpose_to_(enc, b->qw, H, H, 3 * H, 0, _ab_wt[2]) ||
          !transpose_to_(enc, b->kw, H, H, 3 * H, H, _ab_wt[2]) ||
          !transpose_to_(enc, b->vw, H, H, 3 * H, 2 * H, _ab_wt[2])) {
        return fail("qwen-image-2.1 trainer: a weight is not readable");
      }
      enc.end();
      std::string ge;
      const auto tw0 = std::chrono::steady_clock::now();
      if (!st.commit().wait_ok(&ge)) {
        return fail("qwen-image-2.1 train: " + ge);
      }
      _t_start_wait += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - tw0).count();
      st = mc->make_command_stream();
      enc = st.begin_compute();
      ane_back_stage_(abt[0], L);
    }

    // ---- backward: the feed-forward half -----------------------------
    for (const Band& bd : bands) {
      _ops.gated_tanh_bwd(enc, _dres, (std::size_t)bd.start * H, _modb,
                          (std::size_t)bd.mod_row * 4 * H + 3 * H, _da,
                          (std::size_t)bd.start * H, bd.rows, H);
    }
    // ds = da W_down; W_down is [H, FF], its transpose [FF, H].
    if (ab) {
      if (!back_split_(abt[0], L, st, enc, _da, _ds, ROWS, err)) {
        return false;
      }
      ane_back_stage_(abt[1], L);
    } else {
      if (!transpose_in_(enc, b->out_w, H, FF, H, 0)) {
        return fail("qwen-image-2.1 trainer: a down weight is not "
                    "readable");
      }
      mark("bw.transpose");
      back_gemm_(enc, _da, ROWS, H, FF, _ds);
    }
    mark("bw.gemm.down");
    lora_bwd_(enc, L, &BlockLora::out, V.s, 0, FF, _da, 0, H, _ds, 0, FF,
              ROWS, H, FF);
    mark("bw.lora");
    _ops.swiglu_bwd(enc, V.g, V.u, _ds, _dgu, 0, _dgu, (std::size_t)FF, ROWS,
                    FF, 2 * FF);
    // dh2 = [dG | dU] [W_gate ; W_proj]: W^T [H, 2FF].
    if (ab) {
      if (!back_split_(abt[1], L, st, enc, _dgu, _dh, ROWS, err)) {
        return false;
      }
      ane_back_stage_(abt[2], L);
    } else {
      if (!transpose_in_(enc, b->gate_w, FF, H, 2 * FF, 0) ||
          !transpose_in_(enc, b->proj_w, FF, H, 2 * FF, FF)) {
        return fail("qwen-image-2.1 trainer: a gate/up weight is not "
                    "readable");
      }
      mark("bw.transpose");
      back_gemm_(enc, _dgu, ROWS, 2 * FF, H, _dh);
    }
    mark("bw.gemm.gateup");
    lora_bwd_(enc, L, &BlockLora::gate, V.h2, 0, H, _dgu, 0, 2 * FF, _dh, 0,
              H, ROWS, FF, H);
    lora_bwd_(enc, L, &BlockLora::proj, V.h2, 0, H, _dgu, (std::size_t)FF,
              2 * FF, _dh, 0, H, ROWS, FF, H);
    mark("bw.lora");
    for (const Band& bd : bands) {
      const std::size_t o = (std::size_t)bd.start * H;
      _ops.ln_mod_bwd(enc, V.x1, o, _dh, o, false, _modb,
                      (std::size_t)bd.mod_row * 4 * H + 2 * H, _dres, o,
                      bd.rows, H, eps);
    }

    // ---- backward: the attention half --------------------------------
    for (const Band& bd : bands) {
      _ops.gated_tanh_bwd(enc, _dres, (std::size_t)bd.start * H, _modb,
                          (std::size_t)bd.mod_row * 4 * H + H, _da,
                          (std::size_t)bd.start * H, bd.rows, H);
    }
    if (!transpose_in_(enc, b->ow, H, H, H, 0)) {
      return fail("qwen-image-2.1 trainer: an out weight is not readable");
    }
    back_gemm_(enc, _da, ROWS, H, H, _daf);
    mark("bw.gemm.o");
    lora_bwd_(enc, L, &BlockLora::o, V.af, 0, H, _da, 0, H, _daf, 0, H, ROWS,
              H, H);
    mark("bw.lora");
    transpose(_daf, _do, ROWS, NH);
    _ops.fill_f32(enc, _dk32, 0, 0.0f, RH);
    _ops.fill_f32(enc, _dv32, 0, 0.0f, RH);
    _ops.attn_backward(enc, V.qh, V.kh, V.vh, V.ao, _do, V.mlm, V.mls, _lse2,
                       _dsum, _dqh, _dk32, _dv32, NH, ROWS, Hd, ascale, segs);
    mark("bw.attn");
    _ops.rope_bwd(enc, _dqh, false, _dqn, _rcos, _rsin, 0, NH, ROWS, Hd);
    _ops.rope_bwd(enc, _dk32, true, _dkn, _rcos, _rsin, 0, NH, ROWS, Hd);
    // [dq | dk | dv], stacked for one GEMM.
    _ops.rms_bwd(enc, V.qpre, 0, _dqn, 0, b->nq, _dqkv, 0, ROWS * NH, Hd,
                 eps, NH, 3 * H);
    _ops.rms_bwd(enc, V.kpre, 0, _dkn, 0, b->nk, _dqkv, (std::size_t)H,
                 ROWS * NH, Hd, eps, NH, 3 * H);
    _ops.hd_f32_to_td(enc, _dv32, _dqkv, (std::size_t)2 * H, NH, ROWS, Hd,
                      3 * H);
    if (ab) {
      if (!back_split_(abt[2], L, st, enc, _dqkv, _dh, ROWS, err)) {
        return false;
      }
    } else {
      if (!transpose_in_(enc, b->qw, H, H, 3 * H, 0) ||
          !transpose_in_(enc, b->kw, H, H, 3 * H, H) ||
          !transpose_in_(enc, b->vw, H, H, 3 * H, 2 * H)) {
        return fail("qwen-image-2.1 trainer: a q/k/v weight is not "
                    "readable");
      }
      mark("bw.elt+transpose");
      back_gemm_(enc, _dqkv, ROWS, 3 * H, H, _dh);
    }
    mark("bw.gemm.qkv");
    lora_bwd_(enc, L, &BlockLora::q, V.h1, 0, H, _dqkv, 0, 3 * H, _dh, 0, H,
              ROWS, H, H);
    lora_bwd_(enc, L, &BlockLora::k, V.h1, 0, H, _dqkv, (std::size_t)H,
              3 * H, _dh, 0, H, ROWS, H, H);
    lora_bwd_(enc, L, &BlockLora::v, V.h1, 0, H, _dqkv, (std::size_t)2 * H,
              3 * H, _dh, 0, H, ROWS, H, H);
    mark("bw.lora");
    for (const Band& bd : bands) {
      const std::size_t o = (std::size_t)bd.start * H;
      _ops.ln_mod_bwd(enc, _ckpt, x0e + o, _dh, o, false, _modb,
                      (std::size_t)bd.mod_row * 4 * H, _dres, o, bd.rows, H,
                      eps);
    }

    mark("bw.elt");
    // Streamed: the next block down is read under this one's GPU work,
    // into the slot this block is not using.
    enc.end();
    CommandStream::Fence fence = st.commit();
    if (!held && L > 0) {
      const bool nh = (L - 1) < (int)m._blocks.size() &&
                      !m._blocks[(std::size_t)(L - 1)].qw.empty();
      if (!nh) { m._slots.prefetch(L - 1); }
    }
    std::string ge;
    const auto te0 = std::chrono::steady_clock::now();
    if (!fence.wait_ok(&ge)) { return fail("qwen-image-2.1 train: " + ge); }
    _t_end_wait += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - te0).count();
  }

  const auto t_b1 = std::chrono::steady_clock::now();
  if (std::getenv("VPIPE_QWEN_IMAGE21_TRAIN_TIMES") != nullptr &&
      m._mc->session() != nullptr) {
    m._mc->session()->info(fmt(
        "qwen-image-2.1 train times: backward {:.0f} ms = start-wait {:.0f}"
        " + drains {:.0f} + stage-joins {:.0f} + gpu-halves {:.0f} + "
        "finishes {:.0f} + end-waits {:.0f} (+ encode)",
        std::chrono::duration<double, std::milli>(t_b1 - t_f1).count(),
        _t_start_wait, _t_drain, _t_join, _t_gpu_half, _t_finish,
        _t_end_wait));
  }
  _t_start_wait = _t_drain = _t_join = _t_gpu_half = _t_finish =
      _t_end_wait = 0.0;
  if (_prof && m._mc->session() != nullptr) {
    double tot = 0.0;
    for (const auto& bk : _pbuckets) { tot += bk.second; }
    std::string line = fmt("qwen-image-2.1 train profile: joint {}, "
                           "backward {:.0f} ms accounted", ROWS, tot)();
    for (const auto& bk : _pbuckets) {
      line += fmt("; {} {:.0f} ({:.1f}%)", bk.first, bk.second,
                  tot > 0.0 ? 100.0 * bk.second / tot : 0.0)();
    }
    m._mc->session()->info(fmt("{}", line));
    _pbuckets.clear();
  }
  if (m._fp8_failed) {
    m._fp8_failed = false;
    return fail("qwen-image-2.1 trainer: an FP8 weight could not be "
                "widened");
  }
  if (out != nullptr) {
    out->loss = loss;
    out->forward_ms =
        std::chrono::duration<double, std::milli>(t_f1 - t_f0).count();
    out->backward_ms =
        std::chrono::duration<double, std::milli>(t_b1 - t_f1).count();
    if (keep_pred) { out->pred = std::move(pred); }
  }
  return true;
}

}  // namespace genai
}  // namespace vpipe
