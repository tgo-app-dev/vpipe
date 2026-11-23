#include "generative-models/minimax-h3/minimax-h3-trainer.h"

#include "common/vpipe-format.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace vpipe {
namespace genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;
namespace h3 = minimax_h3;

namespace {

// The steel flash-attention parameter block, as attn_steel.metal reads it.
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

std::string
blk_(const char* prefix, int i, const char* rest)
{
  return std::string(prefix) + std::to_string(i) + "." + rest;
}

// The projections of a block: qkv, out, fc1, fc2 (MiniMaxH3Trainer::
// trained_'s index).
constexpr int kNumProj = 4;

double
ms_since_(std::chrono::steady_clock::time_point t0)
{
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

std::string
MiniMaxH3Trainer::module_name(bool refiner, int L, const char* leaf)
{
  return (refiner ? blk_("token_refiner.refiner_blocks.", L, leaf)
                  : blk_("transformer_blocks.", L, leaf));
}

std::unique_ptr<MiniMaxH3Trainer>
MiniMaxH3Trainer::create(MetalMiniMaxH3Transformer* model, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return std::unique_ptr<MiniMaxH3Trainer>();
  };
  if (model == nullptr) { return fail("minimax-h3 trainer: no model"); }
  const M::Config& c = model->_cfg;
  if (c.head_dim != 128) {
    return fail("minimax-h3 trainer: the attention backward is head_dim "
                "128 only");
  }
  if (c.tp.active()) {
    return fail("minimax-h3 trainer: a rank of a split DiT does not train");
  }
  if (!model->_steel_ok) {
    return fail("minimax-h3 trainer: the steel flash attention is off "
                "(VPIPE_H3_NO_STEEL_ATTN or no kernel)");
  }
  if (!model->_fn_gemm_acc_band.valid()) {
    return fail("minimax-h3 trainer: this build has no banded adapter "
                "GEMM, which the fused q|k|v adapter rides");
  }
  std::unique_ptr<MiniMaxH3Trainer> t(new MiniMaxH3Trainer());
  t->_m = model;
  // A preview's denoise would bake the AdaLN and retire what a step reads.
  model->hold_adaln_unbaked(true);
  if (const char* e = std::getenv("VPIPE_H3_TRAIN_I8_BWD")) {
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
      return fail("minimax-h3 trainer: the dequant kernels for group " +
                  std::to_string(model->_quant_group) + " did not load");
    }
  }
  return t;
}

MiniMaxH3Trainer::~MiniMaxH3Trainer()
{
  if (_pf.valid()) { (void)_pf.get(); }
  if (_m != nullptr && _slot >= 0) { _m->release_live_lora(_slot); }
  if (_m != nullptr) { _m->hold_adaln_unbaked(false); }
}

std::vector<train::ModuleShape>
MiniMaxH3Trainer::modules(const std::string& targets, bool refiner) const
{
  const M::Config& c = _m->_cfg;
  const int H = c.hidden, I = c.inner(), F = c.ffn, HD = c.head_dim;
  const bool ff = targets != "attn";
  // The forward's row order, relative to each module's own place: q, k
  // and v are ONE fused B in the base's grouping (k's and v's rows land
  // among q's, before their own offsets), fc1 gate first where diffusers'
  // file is [value; gate].
  std::vector<int> qkv_rows[3], fc1_rows((std::size_t)(2 * F));
  for (int pt = 0; pt < 3; ++pt) {
    qkv_rows[pt].resize((std::size_t)I);
    for (int i = 0; i < I; ++i) {
      qkv_rows[pt][(std::size_t)i] =
          M::qkv_fused_row(pt, i, I, HD, c.qkv_per_head) - pt * I;
    }
  }
  for (int i = 0; i < 2 * F; ++i) {
    fc1_rows[(std::size_t)i] = i < F ? i + F : i - F;
  }
  std::vector<train::ModuleShape> v;
  auto add = [&](bool rf, int L) {
    v.push_back({module_name(rf, L, "attn.to_q"), I, H, qkv_rows[0]});
    v.push_back({module_name(rf, L, "attn.to_k"), I, H, qkv_rows[1]});
    v.push_back({module_name(rf, L, "attn.to_v"), I, H, qkv_rows[2]});
    v.push_back({module_name(rf, L, "attn.to_out.0"), H, I, {}});
    if (ff) {
      v.push_back({module_name(rf, L, "ff.net.0.proj"), 2 * F, H,
                   fc1_rows});
      v.push_back({module_name(rf, L, "ff.net.2"), H, F, {}});
    }
  };
  for (int L = 0; L < c.n_layers; ++L) { add(false, L); }
  if (refiner) {
    for (int R = 0; R < c.n_refiner; ++R) { add(true, R); }
  }
  return v;
}

MiniMaxH3Trainer::Trained&
MiniMaxH3Trainer::trained_(bool refiner, int L,
                           LoraFactors BlockLora::* which)
{
  const int base = refiner ? _m->_cfg.n_layers + L : L;
  const int j = which == &BlockLora::qkv   ? 0
                : which == &BlockLora::out ? 1
                : which == &BlockLora::fc1 ? 2
                                           : 3;
  return _tr[(std::size_t)base * kNumProj + (std::size_t)j];
}

int
MiniMaxH3Trainer::bind(train::LoraParams* params, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "minimax-h3 trainer: " + m; }
    return 0;
  };
  if (params == nullptr) { return fail("no parameters"); }
  if (_slot >= 0) { return fail("an adapter is bound already"); }
  const M::Config& c = _m->_cfg;
  const int H = c.hidden, I = c.inner(), F = c.ffn, HD = c.head_dim;
  const int r = params->rank();
  const bool per_head = c.qkv_per_head;
  std::vector<train::LoraParams::Module>& mods = params->modules();
  auto find = [&](const std::string& nm) -> train::LoraParams::Module* {
    for (train::LoraParams::Module& m : mods) {
      if (m.name == nm) { return &m; }
    }
    return nullptr;
  };
  _tr.assign((std::size_t)(c.n_layers + c.n_refiner) * kNumProj, Trained{});
  _train_refiner = false;
  // The row maps, and every projection's place in the arenas.
  for (int pass = 0; pass < 2; ++pass) {
    const bool rf = pass == 1;
    const int nb = rf ? c.n_refiner : c.n_layers;
    for (int L = 0; L < nb; ++L) {
      train::LoraParams::Module* q = find(module_name(rf, L, "attn.to_q"));
      train::LoraParams::Module* k = find(module_name(rf, L, "attn.to_k"));
      train::LoraParams::Module* v = find(module_name(rf, L, "attn.to_v"));
      if ((q != nullptr) != (k != nullptr) ||
          (q != nullptr) != (v != nullptr)) {
        return fail(fmt("block {}'s q, k and v are trained together or not "
                        "at all", L)());
      }
      if (q != nullptr) {
        // One banded factor: A stacked [3r, H], B [3I, r] in the base's
        // fused row order -- both contiguous in the arenas.
        if (q->n != I || q->k != H || k->n != I || k->k != H ||
            v->n != I || v->k != H ||
            k->a_off != q->a_off + (std::size_t)r * H ||
            v->a_off != k->a_off + (std::size_t)r * H ||
            k->b_off != q->b_off + (std::size_t)I * r ||
            v->b_off != k->b_off + (std::size_t)I * r) {
          return fail(fmt("block {}'s q, k and v are not adjacent [{}, {}] "
                          "modules", L, I, H)());
        }
        const int row0 = (int)(q->b_off / (std::size_t)r);
        train::LoraParams::Module* part[3] = {q, k, v};
        for (int pt = 0; pt < 3; ++pt) {
          part[pt]->b_rows.resize((std::size_t)I);
          for (int i = 0; i < I; ++i) {
            part[pt]->b_rows[(std::size_t)i] =
                row0 + M::qkv_fused_row(pt, i, I, HD, per_head);
          }
        }
        Trained& t = trained_(rf, L, &BlockLora::qkv);
        t.on = true;
        t.a_off = q->a_off;
        t.b_off = q->b_off;
        t.banded = true;
        if (rf) { _train_refiner = true; }
      }
      if (train::LoraParams::Module* o =
              find(module_name(rf, L, "attn.to_out.0"))) {
        if (o->n != H || o->k != I) { return fail(o->name + ": shape"); }
        Trained& t = trained_(rf, L, &BlockLora::out);
        t.on = true; t.a_off = o->a_off; t.b_off = o->b_off;
        if (rf) { _train_refiner = true; }
      }
      if (train::LoraParams::Module* g =
              find(module_name(rf, L, "ff.net.0.proj"))) {
        if (g->n != 2 * F || g->k != H) { return fail(g->name + ": shape"); }
        // diffusers' [value; gate] in the file, [gate | up] here.
        const int row0 = (int)(g->b_off / (std::size_t)r);
        g->b_rows.resize((std::size_t)(2 * F));
        for (int i = 0; i < 2 * F; ++i) {
          g->b_rows[(std::size_t)i] = row0 + (i < F ? i + F : i - F);
        }
        Trained& t = trained_(rf, L, &BlockLora::fc1);
        t.on = true; t.a_off = g->a_off; t.b_off = g->b_off;
        if (rf) { _train_refiner = true; }
      }
      if (train::LoraParams::Module* d =
              find(module_name(rf, L, "ff.net.2"))) {
        if (d->n != H || d->k != F) { return fail(d->name + ": shape"); }
        Trained& t = trained_(rf, L, &BlockLora::fc2);
        t.on = true; t.a_off = d->a_off; t.b_off = d->b_off;
        if (rf) { _train_refiner = true; }
      }
    }
  }
  _p = params;
  const SharedBuffer& as = params->a_shadow();
  const SharedBuffer& bs = params->b_shadow();
  const int qkv_group = per_head ? HD : I;
  auto parse = [&](const std::string& mod, bool* rf, int* L,
                   std::string* leaf) {
    std::string rest;
    if (mod.rfind("blocks.", 0) == 0) {
      *rf = false;
      rest = mod.substr(7);
    } else if (mod.rfind("token_refiner.blocks.", 0) == 0) {
      *rf = true;
      rest = mod.substr(21);
    } else {
      return false;
    }
    const std::size_t dot = rest.find('.');
    if (dot == std::string::npos) { return false; }
    *L = std::atoi(rest.substr(0, dot).c_str());
    *leaf = rest.substr(dot + 1);
    return true;
  };
  int slot = -1;
  std::string berr;
  const int bound = _m->bind_live_lora(
      [&](const std::string& mod, int n, int k, LoraFactors* f) {
        bool rf = false;
        int L = 0;
        std::string leaf;
        if (!parse(mod, &rf, &L, &leaf)) { return false; }
        LoraFactors BlockLora::* which =
            leaf == "attn.qkv_proj"  ? &BlockLora::qkv
            : leaf == "attn.out_proj" ? &BlockLora::out
            : leaf == "mlp.fc1"       ? &BlockLora::fc1
            : leaf == "mlp.fc2"       ? &BlockLora::fc2
                                      : nullptr;
        if (which == nullptr) { return false; }
        const Trained& t = trained_(rf, L, which);
        if (!t.on) { return false; }
        if (which == &BlockLora::qkv) {
          f->rank = 3 * r;
          f->parts = 3;
          f->group = qkv_group;
          f->a = as.subview(t.a_off * 2, (std::size_t)3 * r * k * 2);
          f->b = bs.subview(t.b_off * 2, (std::size_t)n * r * 2);
        } else {
          f->rank = r;
          f->a = as.subview(t.a_off * 2, (std::size_t)r * k * 2);
          f->b = bs.subview(t.b_off * 2, (std::size_t)n * r * 2);
        }
        return !f->a.empty() && !f->b.empty();
      },
      &slot, &berr);
  int want = 0;
  for (const Trained& t : _tr) { want += t.on ? 1 : 0; }
  if (bound == 0 || bound != want) {
    if (slot >= 0) { _m->release_live_lora(slot); }
    _p = nullptr;
    return fail(berr.empty() ? fmt("bound {} of {} projections", bound,
                                   want)()
                             : berr);
  }
  _slot = slot;
  return (int)params->modules().size();
}

std::size_t
MiniMaxH3Trainer::arena_bytes(const MetalMiniMaxH3Transformer::Config& c,
                              int rows, int n_text, int n_t, int rank,
                              bool keep_all)
{
  const std::size_t S = (std::size_t)std::max(rows, n_text);
  const std::size_t T = (std::size_t)n_text;
  const std::size_t H = (std::size_t)c.hidden, I = (std::size_t)c.inner();
  const std::size_t F = (std::size_t)c.ffn, NH = (std::size_t)c.n_heads;
  const std::size_t NL = (std::size_t)c.n_layers;
  const std::size_t NR = (std::size_t)c.n_refiner;
  const std::size_t R = (std::size_t)std::max(3 * rank, 1);
  std::size_t b = 0;
  b += NL * S * H * 2;                                 // checkpoints
  b += NL * (std::size_t)n_t * c.adaln_out() * 2;      // modulations
  b += (std::size_t)n_t * 2 * H * 2 + S * H * 2;       // final
  b += (NR + 1) * T * H * 2;                           // refiner taps
  // one block's recompute -- the refiner's alone where every main block
  // keeps its own (recompute: never)
  const std::size_t SR = keep_all ? std::max<std::size_t>(T, 1) : S;
  b += SR * H * 2 * 3 + S * H * 2;                     // h1, x1, h2; ao
  b += SR * 3 * I * 2 + S * 3 * I * 2;                 // qkv pre; normed
  b += SR * I * 2 * 5;                                 // q, k, v, o, ob
  b += SR * 2 * F * 2 + SR * F * 2;                    // gu, swiglu
  b += NH * SR * 4 * 2 + NH * S * 4 * 2;               // ml; lse, dsum
  if (keep_all) { b += NL * keep_block_bytes(c, (int)S); }
  // the backward
  b += S * H * 4 + T * H * 4;                          // dres (+ text)
  b += S * H * 2 * 3;                                  // df, dh, dproj
  b += S * F * 2 + S * 2 * F * 2;                      // ds, dgu
  b += S * I * 2 * 3;                                  // dob, doh, dq
  b += S * I * 4 * 2;                                  // dk, dv (f32)
  b += 2 * F * H * 2 * 2;                              // W^T, dequant
  b += S * R * 2 * 2;                                  // u, t
  b += 3 * I * R * (2 + 4);                            // banded B, dB
  b += train::TrainOps::attn_scratch_bytes((int)NH, (int)S, (int)S);
  // the forward's rows (x_t, video and audio) and its index tables
  b += S * ((std::size_t)c.video_patch_elems() + c.audio_channels) * 2;
  b += S * 4 * 2;
  return b;
}

bool
MiniMaxH3Trainer::ensure_arena_(int seq, int n_text, int n_t)
{
  if (seq <= _seq && n_text <= _ntext && n_t <= _nt) { return true; }
  metal_compute::MetalCompute* mc = _m->_mc;
  const M::Config& c = _m->_cfg;
  seq = std::max(seq, _seq);
  n_text = std::max(n_text, _ntext);
  n_t = std::max(n_t, _nt);
  const std::size_t S = (std::size_t)std::max(seq, n_text);
  const std::size_t T = (std::size_t)n_text;
  const std::size_t H = (std::size_t)c.hidden, I = (std::size_t)c.inner();
  const std::size_t F = (std::size_t)c.ffn, NH = (std::size_t)c.n_heads;
  const std::size_t NL = (std::size_t)c.n_layers;
  const std::size_t NR = (std::size_t)c.n_refiner;
  int rmax = 1;
  for (const M::LoraSlot& sl : _m->_lora) {
    rmax = std::max(rmax, sl.max_rank);
  }
  const std::size_t R = (std::size_t)rmax;
  auto bf = [&](std::size_t e) { return mc->make_shared_buffer(e * 2); };
  auto f32 = [&](std::size_t e) { return mc->make_shared_buffer(e * 4); };
  _ckpt = bf(NL * (std::size_t)seq * H);
  _mods = bf(NL * (std::size_t)n_t * (std::size_t)c.adaln_out());
  _fmod = bf((std::size_t)n_t * 2 * H);
  _final = bf((std::size_t)seq * H);
  _ref_in = bf(std::max<std::size_t>(NR, 1) * T * H);
  _ref_out = bf(T * H);
  // The recompute's activations: every block's, or -- with each main
  // block's kept (recompute: never) -- the token refiner's alone.
  const std::size_t SR = _keep_all ? std::max<std::size_t>(T, 1) : S;
  _h1 = bf(SR * H); _ao = bf(S * H); _x1 = bf(SR * H); _h2 = bf(SR * H);
  _qkvp = bf(SR * 3 * I); _qkvn = bf(S * 3 * I);
  _qh = bf(SR * I); _kh = bf(SR * I); _vh = bf(SR * I); _oh = bf(SR * I);
  _ob = bf(SR * I);
  _gu = bf(SR * 2 * F); _sw = bf(SR * F);
  _mlm = f32(NH * SR); _mls = f32(NH * SR);
  _keep.clear();
  if (_keep_all) {
    _keep_rows = (int)S;
    const std::size_t kb = keep_block_bytes(c, (int)S);
    for (std::size_t l = 0; l < NL; ++l) {
      _keep.push_back(mc->make_shared_buffer(kb));
      if (_keep.back().empty()) {
        _keep.clear();
        _seq = _ntext = _nt = 0;
        return false;
      }
    }
  }
  _lse2 = f32(NH * S); _dsum = f32(NH * S);
  _dres = f32(S * H); _dres_t = f32(T * H);
  _df = bf(S * H); _dh = bf(S * H); _dproj = bf(S * H);
  _ds = bf(S * F); _dgu = bf(S * 2 * F);
  _dob = bf(S * I); _doh = bf(S * I); _dqh = bf(S * I);
  _dk32 = f32(S * I); _dv32 = f32(S * I);
  _wt = bf(std::max(2 * F * H, 3 * I * H));
  _deq = _m->_quant_bits > 0 ? bf(std::max(2 * F * H, 3 * I * H))
                             : SharedBuffer{};
  _lu = bf(S * R); _lt = bf(S * R);
  _bbd = bf(std::max(3 * I, 2 * F) * R);
  _dbbd = f32(3 * I * R);
  _zero = bf(std::max<std::size_t>(6 * H, 64));
  const std::size_t VPE = (std::size_t)c.video_patch_elems();
  _dpv = bf(S * VPE);
  _dpa = bf(S * (std::size_t)c.audio_channels);
  _part = f32((S * std::max<std::size_t>(VPE, 32)) / 256 + 64);
  const SharedBuffer* all[] = {
      &_ckpt, &_mods, &_fmod, &_final, &_ref_in, &_ref_out, &_h1, &_ao,
      &_x1, &_h2, &_qkvp, &_qkvn, &_qh, &_kh, &_vh, &_oh, &_ob, &_gu, &_sw,
      &_mlm, &_mls, &_lse2, &_dsum, &_dres, &_dres_t, &_df, &_dh, &_dproj,
      &_ds, &_dgu, &_dob, &_doh, &_dqh, &_dk32, &_dv32, &_wt, &_lu, &_lt,
      &_bbd, &_dbbd, &_zero, &_dpv, &_dpa, &_part};
  for (const SharedBuffer* b : all) {
    if (b->empty()) {
      _seq = _ntext = _nt = 0;
      return false;
    }
  }
  if (_m->_quant_bits > 0 && _deq.empty()) {
    _seq = _ntext = _nt = 0;
    return false;
  }
  std::memset(_zero.contents(), 0, _zero.byte_size());
  _seq = seq;
  _ntext = n_text;
  _nt = n_t;
  return true;
}

const SharedBuffer*
MiniMaxH3Trainer::dense_(ComputeEncoder& enc, const Linear& l, int N, int K,
                         Linear* view)
{
  M& m = *_m;
  if (l.gc > 0) {
    const SharedBuffer* codes = &l.codes;
    const SharedBuffer* scales = &l.cscales;
    if (l.cr_w4) {
      const std::size_t n = (std::size_t)N * (std::size_t)K;
      if (m._cr_codes.empty() || m._cr_codes.byte_size() < n) {
        m._cr_codes = m._mc->make_shared_buffer(n);
        if (m._cr_codes.empty()) { return nullptr; }
      }
      const bool cb = !l.f8_cb.empty();
      enc.set_function(m._fn_w4a8_i8);
      enc.set_buffer(0, l.codes);
      enc.set_buffer(1, m._cr_codes);
      enc.set_constant(2, (std::uint32_t)n);
      enc.set_buffer(3, l.f8_srel);
      enc.set_buffer(4, cb ? l.f8_cb : l.cscales);
      enc.set_constant(5, (std::uint32_t)l.f8_qgroup);
      enc.set_constant(6, (std::uint32_t)(cb ? 1 : 0));
      enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
      codes = &m._cr_codes;
    }
    if (l.pc) {
      if (!m._i8 || !m._i8->encode_to_fast_scratch(enc, l.codes, l.scales,
                                                   l.qbias, l.gc, N, K)) {
        return nullptr;
      }
      codes = &m._i8->fast_scratch_codes();
      scales = &m._i8->fast_scratch_scales();
    }
    const Linear* d = m.i8_dense_(enc, l, *codes, *scales, *view, N, K);
    return d != nullptr ? &d->w : nullptr;
  }
  if (l.is_fp8()) {
    const Linear* d = m.fp8_dense_(enc, l, *view, N, K);
    return d != nullptr ? &d->w : nullptr;
  }
  if (!l.quantized) { return &l.w; }
  const metal_compute::ComputeFunction& dq =
      l.bits == 8 ? _fn_deq8 : _fn_deq4;
  if (!dq.valid() || _deq.empty()) { return nullptr; }
  enc.set_function(dq);
  enc.set_buffer(0, l.codes); enc.set_buffer(1, l.scales);
  enc.set_buffer(2, l.qbias); enc.set_buffer(3, _deq);
  enc.set_constant(4, K); enc.set_constant(5, N);
  const unsigned words = (unsigned)(l.bits == 8 ? (K / 4) : (K / 8));
  enc.dispatch({words, (unsigned)N, 1}, {64, 1, 1});
  return &_deq;
}

bool
MiniMaxH3Trainer::back_gemm_(ComputeEncoder& enc, const Linear& l, int N,
                             int K, const SharedBuffer& dy, std::size_t dye,
                             const SharedBuffer& dx, std::size_t dxe, int M)
{
  Linear view;
  const SharedBuffer* d = dense_(enc, l, N, K, &view);
  if (d == nullptr) { return false; }
  // W^T [K, N], then dX [M, K] = dY [M, N] (W^T)^T through the forward's
  // own route: the scratch is an ordinary dense [K, N] weight.
  _ops.transpose(enc, *d, 0, _wt, 0, N, K, K, N, 0);
  Linear wt;
  wt.w = _wt.subview(0, (std::size_t)K * N * 2);
  _m->_i8_hold = !_i8_bwd;
  _m->gemm_(enc, dy, dye, wt, dx, dxe, M, K, N);
  _m->_i8_hold = false;
  return true;
}

void
MiniMaxH3Trainer::lora_bwd_(ComputeEncoder& enc, bool refiner, int L,
                            LoraFactors BlockLora::* which,
                            const SharedBuffer& x, int ldx,
                            const SharedBuffer& dy, int ldy,
                            const SharedBuffer& dx, int lddx, int Mr, int N,
                            int K)
{
  M& m = *_m;
  for (std::size_t si = 0; si < m._lora.size(); ++si) {
    const M::LoraSlot& sl = m._lora[si];
    const std::vector<BlockLora>& v = refiner ? sl.refiner : sl.blocks;
    if (L < 0 || (std::size_t)L >= v.size()) { continue; }
    const LoraFactors* f = &(v[(std::size_t)L].*which);
    // Spilled with its block: the copy fill_lora_ read for this block.
    if (f->empty() && f->rank > 0 && !refiner &&
        L == m._lora_stream_layer && si < m._lora_stream.size()) {
      f = &(m._lora_stream[si].*which);
    }
    const float s = sl.scale;
    if (f->empty() || s == 0.0f) { continue; }
    const int R = f->rank;
    // A banded B [N, R / parts] as the block-diagonal [N, R] its GEMMs read.
    const SharedBuffer* bop = &f->b;
    if (f->banded()) {
      _ops.band_expand(enc, f->b, 0, _bbd, N, f->b_rank(), f->parts,
                       f->group);
      bop = &_bbd;
    }
    // u = dY B  [M, R]
    _ops.gemm_nn(enc, dy, 0, *bop, 0, _lu, 0, Mr, R, N, ldy, R, R, 1.0f,
                 false);
    const Trained* t =
        (int)si == _slot ? &trained_(refiner, L, which) : nullptr;
    if (t != nullptr && t->on) {
      const float c = _p->a_mul();
      // t = x A^T  [M, R], A the shadow (alpha / rank folded in).
      _ops.gemm_nt(enc, x, 0, f->a, 0, _lt, 0, Mr, R, K, ldx, K, R, 1.0f,
                   false);
      // dB += s dY^T t: straight into the arena, or -- banded -- the
      // block-diagonal product, its band folded back.
      if (t->banded) {
        _ops.gemm_tn_f32(enc, dy, 0, _lt, 0, _dbbd, 0, Mr, N, R, ldy, R, R,
                         s, false);
        _ops.band_extract_acc(enc, _dbbd, _p->b_grad(), t->b_off, N,
                              f->b_rank(), f->parts, f->group, 1.0f);
      } else {
        _ops.gemm_tn_f32(enc, dy, 0, _lt, 0, _p->b_grad(), t->b_off, Mr, N,
                         R, ldy, R, R, s, true);
      }
      // dA += c s u^T x
      _ops.gemm_tn_f32(enc, _lu, 0, x, 0, _p->a_grad(), t->a_off, Mr, R, K,
                       R, ldx, K, c * s, true);
    }
    // dX += s u A
    _ops.gemm_nn(enc, _lu, 0, f->a, 0, dx, 0, Mr, K, R, R, K, lddx, s, true);
  }
}

const MiniMaxH3Trainer::Block*
MiniMaxH3Trainer::acquire_(int L, Block& local, std::string* err)
{
  M& m = *_m;
  if (L < (int)m._blocks.size() && !m._blocks[(std::size_t)L].qkv.empty()) {
    return &m._blocks[(std::size_t)L];
  }
  const std::string prefix = blk_("blocks.", L, "");
  if (m._slots_ready && !m._slots_off) {
    int use = -1;
    bool ok = false;
    if (_pf.valid() && _pf_layer == L) {
      ok = _pf.get();
      use = _pf_slot;
    } else {
      if (_pf.valid()) { (void)_pf.get(); }
      use = m._slot_cur;
      // The backward reads no AdaLN: the forward's modulation is in the
      // tap. A slot built with it keeps the stale bytes, and the next
      // forward's refill rewrites them.
      ok = m.refill_block_(*m._ws, prefix, m._slot[use], false);
    }
    _pf_layer = _pf_slot = -1;
    if (ok) {
      m._slot_cur = use;
      return &m._slot[use];
    }
  }
  if (_pf.valid()) { (void)_pf.get(); }
  _pf_layer = _pf_slot = -1;
  if (!m.load_block_(*m._ws, prefix, local, /*main=*/true,
                     /*with_adaln=*/false, M::Retain::Streamed)) {
    if (err != nullptr) {
      *err = fmt("minimax-h3 trainer: block {} did not load", L)();
    }
    return nullptr;
  }
  return &local;
}

void
MiniMaxH3Trainer::prefetch_(int L)
{
  M& m = *_m;
  if (L < 0 || _pf.valid() || !m._slots_ready || m._slots_off ||
      !m._slot_pair) {
    return;
  }
  if (L < (int)m._blocks.size() && !m._blocks[(std::size_t)L].qkv.empty()) {
    return;
  }
  const int dst = m._slot_cur ^ 1;
  _pf_layer = L;
  _pf_slot = dst;
  _pf = std::async(std::launch::async, [&m, L, dst]() {
    return m.refill_block_(*m._ws, blk_("blocks.", L, ""), m._slot[dst],
                           /*with_adaln=*/false);
  });
}

bool
MiniMaxH3Trainer::inputs_(const Sample& s, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "minimax-h3 trainer: " + m; }
    return false;
  };
  const M::Config& c = _m->_cfg;
  const h3::PackedLayout& L = *s.layout;
  const int VPE = c.video_patch_elems(), AC = c.audio_channels;
  const int nvc = L.num_condition_video_rows, nvg = L.num_video_rows;
  const int nac = L.num_condition_audio_rows, nag = L.num_audio_rows;
  if ((int)L.video_indices.size() != nvc + nvg ||
      (int)L.audio_indices.size() != nac + nag) {
    return fail("the layout's index vectors disagree with its counts");
  }
  if (nvg > 0 && (s.video_x0 == nullptr || s.video_noise == nullptr)) {
    return fail("the layout has generated video rows and no latents");
  }
  if (nag > 0 && (s.audio_x0 == nullptr || s.audio_noise == nullptr)) {
    return fail("the layout has generated audio rows and no latents");
  }
  if ((nvc > 0 && s.video_cond == nullptr) ||
      (nac > 0 && s.audio_cond == nullptr)) {
    return fail("the layout has conditioning rows and none were given");
  }
  auto fill = [&](SharedBuffer& dst, int nc, int ng, int w,
                  const SharedBuffer* cond, const SharedBuffer* x0,
                  const SharedBuffer* nz, float sig) {
    const std::size_t n = (std::size_t)(nc + ng) * w;
    if (n == 0) { dst = SharedBuffer{}; return true; }
    if (dst.byte_size() < n * 2) {
      dst = _m->_mc->make_shared_buffer(n * 2);
      if (dst.empty()) { return false; }
    }
    auto* d = static_cast<std::uint16_t*>(dst.contents());
    if (nc > 0) {
      std::memcpy(d, cond->contents(), (std::size_t)nc * w * 2);
    }
    const auto* a = static_cast<const std::uint16_t*>(x0 != nullptr
                                                          ? x0->contents()
                                                          : nullptr);
    const auto* e = static_cast<const std::uint16_t*>(nz != nullptr
                                                          ? nz->contents()
                                                          : nullptr);
    const std::size_t g = (std::size_t)ng * w;
    d += (std::size_t)nc * w;
    for (std::size_t i = 0; i < g; ++i) {
      d[i] = bf16_((1.0f - sig) * f32_(a[i]) + sig * f32_(e[i]));
    }
    return true;
  };
  if (!fill(_xv, nvc, nvg, VPE, s.video_cond, s.video_x0, s.video_noise,
            s.sigma_video) ||
      !fill(_xa, nac, nag, AC, s.audio_cond, s.audio_x0, s.audio_noise,
            s.sigma_audio)) {
    return fail("input allocation failed");
  }
  h3::build_row_timesteps(L, 1.0f - s.sigma_video, 1.0f - s.sigma_audio,
                          s.cond_video_t, &_uniq, &_row_idx, s.cond_audio_t);
  if ((int)_row_idx.size() != L.seq_len || _uniq.empty()) {
    return fail("the rows' timesteps did not build");
  }
  const std::vector<int> ai = h3::build_adaln_indices(L, _row_idx);
  const std::size_t ib = (std::size_t)L.seq_len * 4;
  if (_aidx.byte_size() < ib) { _aidx = _m->_mc->make_shared_buffer(ib); }
  if (_tidx.byte_size() < ib) { _tidx = _m->_mc->make_shared_buffer(ib); }
  if (_aidx.empty() || _tidx.empty() || (int)ai.size() != L.seq_len) {
    return fail("index allocation failed");
  }
  std::memcpy(_aidx.contents(), ai.data(), ib);
  std::memcpy(_tidx.contents(), _row_idx.data(), ib);
  return true;
}

bool
MiniMaxH3Trainer::loss(const Sample& smp, Result* out, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (smp.layout == nullptr || smp.text == nullptr) {
    return fail("minimax-h3 trainer: an incomplete sample");
  }
  if (!inputs_(smp, err)) { return false; }
  const M::Config& c = _m->_cfg;
  const h3::PackedLayout& L = *smp.layout;
  M::Step st;
  st.video = _xv.empty() ? nullptr : &_xv;
  st.audio = _xa.empty() ? nullptr : &_xa;
  st.text = smp.text;
  st.layout = &L;
  st.timesteps = &_uniq;
  st.row_timestep_index = &_row_idx;
  st.out_head = smp.out_head;
  st.dense_attention = true;
  std::string ferr;
  const M::Velocity v = _m->forward(st, &ferr);
  if (v.empty()) {
    // A stop is not a failure: the forward's own stop check ("stopped")
    // reads here as the empty reason the stage takes for a stop.
    if (ferr.empty() || (_stop && _stop())) {
      if (err != nullptr) { err->clear(); }
      return false;
    }
    return fail(ferr);
  }
  auto mse = [&](const SharedBuffer& pred, int off_rows, int rows, int w,
                 const SharedBuffer* x0, const SharedBuffer* nz) {
    if (rows <= 0) { return 0.0; }
    const auto* p = static_cast<const std::uint16_t*>(pred.contents()) +
                    (std::size_t)off_rows * w;
    const auto* a = static_cast<const std::uint16_t*>(x0->contents());
    const auto* e = static_cast<const std::uint16_t*>(nz->contents());
    const std::size_t n = (std::size_t)rows * w;
    double sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double d =
          (double)f32_(p[i]) - ((double)f32_(a[i]) - (double)f32_(e[i]));
      sq += d * d;
    }
    return sq / (double)n;
  };
  const double lv = mse(v.video, L.num_condition_video_rows,
                        L.num_video_rows, c.video_patch_elems(),
                        smp.video_x0, smp.video_noise);
  const double la = L.num_audio_rows > 0 && smp.audio_weight != 0.0f
                        ? mse(v.audio, L.num_condition_audio_rows,
                              L.num_audio_rows, c.audio_channels,
                              smp.audio_x0, smp.audio_noise)
                        : 0.0;
  if (out != nullptr) {
    out->loss_video = lv;
    out->loss_audio = la;
    out->loss = smp.weight * (lv + smp.audio_weight * la);
  }
  return true;
}

std::size_t
MiniMaxH3Trainer::keep_block_bytes(const MetalMiniMaxH3Transformer::Config& c,
                                   int rows)
{
  const std::size_t S = (std::size_t)std::max(rows, 1);
  const std::size_t H = (std::size_t)c.hidden, I = (std::size_t)c.inner();
  const std::size_t F = (std::size_t)c.ffn, NH = (std::size_t)c.n_heads;
  const std::size_t parts[] = {S * H * 2,     S * H * 2,     S * H * 2,
                               S * 3 * I * 2, S * I * 2,     S * I * 2,
                               S * I * 2,     S * I * 2,     S * I * 2,
                               S * 2 * F * 2, S * F * 2,     NH * S * 4,
                               NH * S * 4};
  std::size_t b = 0;
  for (std::size_t x : parts) { b += (x + 255) & ~(std::size_t)255; }
  return b;
}

MiniMaxH3Trainer::Acts
MiniMaxH3Trainer::scratch_acts_() const
{
  auto v = [](const SharedBuffer& x) { return x.subview(0, x.byte_size()); };
  Acts a;
  a.h1 = v(_h1); a.x1 = v(_x1); a.h2 = v(_h2); a.qkvp = v(_qkvp);
  a.qh = v(_qh); a.kh = v(_kh); a.vh = v(_vh); a.oh = v(_oh); a.ob = v(_ob);
  a.gu = v(_gu); a.sw = v(_sw); a.mlm = v(_mlm); a.mls = v(_mls);
  return a;
}

MiniMaxH3Trainer::Acts
MiniMaxH3Trainer::keep_acts_(int L) const
{
  const M::Config& c = _m->_cfg;
  const SharedBuffer& kb = _keep[(std::size_t)L];
  const std::size_t S = (std::size_t)std::max(_keep_rows, 1);
  const std::size_t H = (std::size_t)c.hidden, I = (std::size_t)c.inner();
  const std::size_t F = (std::size_t)c.ffn, NH = (std::size_t)c.n_heads;
  std::size_t off = 0;
  auto piece = [&](std::size_t bytes) {
    SharedBuffer v = kb.subview(off, bytes);
    off += (bytes + 255) & ~(std::size_t)255;
    return v;
  };
  // The order keep_block_bytes counts them in.
  Acts a;
  a.h1 = piece(S * H * 2); a.x1 = piece(S * H * 2); a.h2 = piece(S * H * 2);
  a.qkvp = piece(S * 3 * I * 2);
  a.qh = piece(S * I * 2); a.kh = piece(S * I * 2); a.vh = piece(S * I * 2);
  a.oh = piece(S * I * 2); a.ob = piece(S * I * 2);
  a.gu = piece(S * 2 * F * 2); a.sw = piece(S * F * 2);
  a.mlm = piece(NH * S * 4); a.mls = piece(NH * S * 4);
  return a;
}

bool
MiniMaxH3Trainer::block_pass_(ComputeEncoder& enc, Pass pass, bool refiner,
                              int L, const Block& b, const SharedBuffer& x,
                              std::size_t xe, int rows,
                              const SharedBuffer& mod, std::size_t me,
                              const SharedBuffer& dres, const Acts& a,
                              std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "minimax-h3 trainer: " + m; }
    return false;
  };
  M& m = *_m;
  const M::Config& c = m._cfg;
  const int H = c.hidden, I = c.inner(), NH = c.n_heads, HD = c.head_dim;
  const int F = c.ffn;
  const bool modulated = !refiner;
  const int QKV_HS = c.qkv_per_head ? 3 * HD : HD;
  const int Q_OFF = 0;
  const int K_OFF = c.qkv_per_head ? HD : I;
  const int V_OFF = c.qkv_per_head ? 2 * HD : 2 * I;
  const int rot = modulated ? c.rope_rot() : 0;
  const int stride6 = 6 * H;
  const float eps = c.norm_eps;
  if (b.fc1.gu_inter) {
    return fail("block " + std::to_string(L) + "'s fc1 is interleaved for "
                "the fused SwiGLU kernel, which keeps no gate / up; load "
                "the model unfused to train it");
  }
  const M::LoraStack lo_qkv = m.lora_stack_(refiner, L, &BlockLora::qkv);
  const M::LoraStack lo_out = m.lora_stack_(refiner, L, &BlockLora::out);
  const M::LoraStack lo_fc1 = m.lora_stack_(refiner, L, &BlockLora::fc1);
  const int elt_h = M::elt_rows(H);
  const int elt_ff = M::elt_rows(F);
  // ---- recompute, keeping what the backward reads ----------------------
  auto rms = [&](const SharedBuffer& in, std::size_t ie,
                 const SharedBuffer& g, const SharedBuffer& y) {
    enc.set_function(m._fn_rms);
    enc.set_buffer(0, in, ie * 2);
    enc.set_buffer(1, g);
    enc.set_buffer(2, y);
    enc.set_constant(3, H);
    enc.set_constant(4, eps);
    enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
  };
  auto modulate = [&](const SharedBuffer& y, int scale_off, int shift_off) {
    enc.set_function(m._fn_modulate);
    enc.set_buffer(1, mod, me * 2);
    enc.set_constant(4, H);
    enc.set_constant(5, stride6);
    enc.set_constant(6, scale_off);
    enc.set_constant(7, shift_off);
    for (int r0 = 0; r0 < rows; r0 += elt_h) {
      const int n = std::min(elt_h, rows - r0);
      const std::size_t eo = (std::size_t)r0 * H * 2;
      enc.set_buffer(0, y, eo);
      enc.set_buffer(2, _aidx, (std::size_t)r0 * 4);
      enc.set_buffer(3, y, eo);
      enc.set_constant(8, n * H);
      enc.dispatch({(unsigned)(n * H), 1, 1}, {256, 1, 1});
    }
  };
  auto add_branch = [&](const SharedBuffer& y, const SharedBuffer& sub,
                        int gate_off) {
    for (int r0 = 0; r0 < rows; r0 += elt_h) {
      const int n = std::min(elt_h, rows - r0);
      const std::size_t eo = (std::size_t)r0 * H * 2;
      if (modulated) {
        enc.set_function(m._fn_gated);
        enc.set_buffer(0, y, eo);
        enc.set_buffer(1, mod, me * 2);
        enc.set_buffer(2, _aidx, (std::size_t)r0 * 4);
        enc.set_buffer(3, sub, eo);
        enc.set_constant(4, H);
        enc.set_constant(5, stride6);
        enc.set_constant(6, gate_off);
        enc.set_constant(7, n * H);
      } else {
        enc.set_function(m._fn_residual);
        enc.set_buffer(0, y, eo);
        enc.set_buffer(1, sub, eo);
        enc.set_buffer(2, y, eo);
        enc.set_constant(3, n * H);
      }
      enc.dispatch({(unsigned)(n * H), 1, 1}, {256, 1, 1});
    }
  };
  auto qk_norm = [&](const SharedBuffer& g, int off) {
    enc.set_function(m._fn_rms_heads);
    enc.set_buffer(0, _qkvn); enc.set_buffer(1, g);
    enc.set_constant(2, rows);
    enc.set_constant(3, NH);
    enc.set_constant(4, HD);
    enc.set_constant(5, 3 * I);
    enc.set_constant(6, off);
    enc.set_constant(7, c.qk_norm_eps);
    enc.set_constant(8, QKV_HS);
    enc.dispatch({32, (unsigned)(rows * NH), 1}, {32, 1, 1});
  };
  // Token-major [rows, ld] (a head every `hs` from `off`) -> head-major
  // [NH, rows, HD], its first `r` dims rotated.
  auto trope = [&](const SharedBuffer& in, const SharedBuffer& out, int ld,
                   int off, int hs, int r) {
    enc.set_function(m._fn_trope);
    enc.set_buffer(0, in); enc.set_buffer(1, out);
    enc.set_buffer(2, _rcos); enc.set_buffer(3, _rsin);
    enc.set_constant(4, NH);
    enc.set_constant(5, rows);
    enc.set_constant(6, HD);
    enc.set_constant(7, r);
    enc.set_constant(8, ld);
    enc.set_constant(9, off);
    enc.set_constant(10, hs);
    enc.dispatch({(unsigned)HD, (unsigned)rows, (unsigned)NH},
                 {(unsigned)HD, 1, 1});
  };
  // [A, B, HD] -> [B, A, HD]
  auto transpose = [&](const SharedBuffer& in, const SharedBuffer& out,
                       int A, int B) {
    enc.set_function(m._fn_transpose);
    enc.set_buffer(0, in); enc.set_buffer(1, out);
    enc.set_constant(2, A); enc.set_constant(3, B);
    enc.set_constant(4, HD);
    enc.dispatch({(unsigned)HD, (unsigned)B, (unsigned)A},
                 {(unsigned)HD, 1, 1});
  };

  // A projection with adapters, in row pieces the forward's [rows, rank]
  // scratch holds: a long sequence's forward runs banded, its scratch a
  // band's, and the recompute runs the whole.
  auto proj = [&](const SharedBuffer& in, const Linear& l,
                  const SharedBuffer& out, int N, int K,
                  const M::LoraStack& lo) {
    const int piece = lo.empty() || m._lora_scratch_rows == 0
                          ? rows
                          : (int)std::min<std::size_t>(
                                (std::size_t)rows, m._lora_scratch_rows);
    for (int r0 = 0; r0 < rows; r0 += piece) {
      const int n = std::min(piece, rows - r0);
      m.gemm_(enc, in, (std::size_t)r0 * K, l, out, (std::size_t)r0 * N, n,
              N, K, lo.empty() ? nullptr : &lo);
    }
  };
  const float ascale = 1.0f / std::sqrt((float)HD);
  if (pass != Pass::kBackward) {
    rms(x, xe, b.n1, a.h1);
    if (modulated) { modulate(a.h1, H, 0); }
    proj(a.h1, b.qkv, a.qkvp, 3 * I, H, lo_qkv);
    enc.set_function(m._fn_copy);
    {
      const std::size_t n = (std::size_t)rows * 3 * I;
      // In pieces under 2^31 elements (copy_f16 counts in an int).
      const std::size_t piece = (std::size_t)1 << 30;
      for (std::size_t o = 0; o < n; o += piece) {
        const std::size_t k = std::min(piece, n - o);
        enc.set_buffer(0, a.qkvp, o * 2);
        enc.set_buffer(1, _qkvn, o * 2);
        enc.set_constant(2, 0);
        enc.set_constant(3, (int)k);
        enc.dispatch({(unsigned)k, 1, 1}, {256, 1, 1});
      }
    }
    qk_norm(b.qn, Q_OFF);
    qk_norm(b.kn, K_OFF);
    trope(_qkvn, a.qh, 3 * I, Q_OFF, QKV_HS, rot);
    trope(_qkvn, a.kh, 3 * I, K_OFF, QKV_HS, rot);
    trope(_qkvn, a.vh, 3 * I, V_OFF, QKV_HS, 0);
    // The attention, its row statistics exported for the backward.
    const bool nax = m._attn_nax;
    const int A_BQ = nax ? 64 : 32;
    const int A_BK = nax ? 32 : 16;
    {
      SteelAttnParams p{};
      p.B = 1; p.H = NH; p.D = HD; p.qL = rows; p.kL = rows;
      p.gqa_factor = 1; p.scale = ascale;
      p.NQ = (rows + A_BQ - 1) / A_BQ; p.NK = (rows + A_BK - 1) / A_BK;
      p.NQ_aligned = rows / A_BQ; p.NK_aligned = rows / A_BK;
      p.qL_rem = rows - p.NQ_aligned * A_BQ;
      p.kL_rem = rows - p.NK_aligned * A_BK;
      p.qL_off = 0;
      const std::int64_t hs = (std::int64_t)rows * HD;
      std::int64_t* const sv[4] = {p.Q_strides, p.K_strides, p.V_strides,
                                   p.O_strides};
      for (std::int64_t* x3 : sv) {
        x3[0] = (std::int64_t)NH * hs; x3[1] = hs; x3[2] = HD;
      }
      metal_compute::FunctionConstants fc;
      fc.set_bool(200, (rows % A_BQ) == 0).set_bool(201, (rows % A_BK) == 0)
          .set_bool(300, false).set_bool(301, false)
          .set_bool(302, false).set_bool(304, true);
      metal_compute::ComputeFunction fn =
          nax ? m._lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc)
              : m._lib_attn.function("attn_steel_h_bd128_bf16", fc);
      if (!fn.valid()) {
        return fail("the exporting attention kernel did not specialize");
      }
      enc.set_function(fn);
      enc.set_buffer(0, a.qh); enc.set_buffer(1, a.kh);
      enc.set_buffer(2, a.vh); enc.set_buffer(3, a.oh);
      enc.set_constant(4, p);
      enc.set_buffer(12, a.mlm); enc.set_buffer(13, a.mls);
      enc.dispatch({(unsigned)(32 * p.NQ), (unsigned)(4 * NH), 1},
                   {32, 4, 1});
      // The meter: the recompute is work the GPU does again.
      m._flops.attention(NH, (long long)rows * rows, HD);
    }
    transpose(a.oh, a.ob, NH, rows);
    proj(a.ob, b.out, _ao, H, I, lo_out);
    // x1 = x + gate1 attn (the refiner: + attn).
    {
      const std::size_t n = (std::size_t)rows * H;
      enc.set_function(m._fn_copy);
      enc.set_buffer(0, x, xe * 2);
      enc.set_buffer(1, a.x1);
      enc.set_constant(2, 0);
      enc.set_constant(3, (int)n);
      enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    }
    add_branch(a.x1, _ao, 2 * H);
    rms(a.x1, 0, b.n2, a.h2);
    if (modulated) { modulate(a.h2, 4 * H, 3 * H); }
    proj(a.h2, b.fc1, a.gu, 2 * F, H, lo_fc1);
    enc.set_function(m._fn_swiglu);
    enc.set_constant(3, F);
    for (int f0 = 0; f0 < rows; f0 += elt_ff) {
      const int fn = std::min(elt_ff, rows - f0);
      enc.set_buffer(0, a.gu, (std::size_t)f0 * 2 * F * 2);
      enc.set_buffer(1, a.sw, (std::size_t)f0 * F * 2);
      enc.set_constant(2, fn);
      enc.dispatch({(unsigned)(fn * F), 1, 1}, {256, 1, 1});
    }
  }
  // FORWARD ONLY (recompute: never): the block's output, x = x1 + gate2
  // fc2(swiglu), into x; its activations stay where `a` put them.
  if (pass == Pass::kForward) {
    const M::LoraStack lo_fc2 = m.lora_stack_(refiner, L, &BlockLora::fc2);
    proj(a.sw, b.fc2, _ao, H, F, lo_fc2);
    const std::size_t n = (std::size_t)rows * H;
    SharedBuffer xo = x.subview(xe * 2, n * 2);
    enc.set_function(m._fn_copy);
    enc.set_buffer(0, a.x1);
    enc.set_buffer(1, xo);
    enc.set_constant(2, 0);
    enc.set_constant(3, (int)n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    add_branch(xo, _ao, 5 * H);
    return true;
  }

  // ---- backward: the feed-forward half -----------------------------
  _ops.gated_idx_bwd(enc, dres, 0, mod, me, _aidx, 0, _df, 0, rows, H,
                     stride6, 5 * H, modulated);
  // ds = df W_fc2 ([H, F])
  if (!back_gemm_(enc, b.fc2, H, F, _df, 0, _ds, 0, rows)) {
    return fail("a down projection is not readable");
  }
  lora_bwd_(enc, refiner, L, &BlockLora::fc2, a.sw, F, _df, H, _ds, F, rows,
            H, F);
  _ops.swiglu_gf_bwd(enc, a.gu, _ds, _dgu, rows, F);
  // dh2 = dgu W_fc1 ([2F, H])
  if (!back_gemm_(enc, b.fc1, 2 * F, H, _dgu, 0, _dh, 0, rows)) {
    return fail("a gate / up projection is not readable");
  }
  lora_bwd_(enc, refiner, L, &BlockLora::fc1, a.h2, H, _dgu, 2 * F, _dh, H,
            rows, 2 * F, H);
  _ops.rms_mod_bwd(enc, a.x1, 0, _dh, 0, false, b.n2, mod, me, _aidx, 0,
                   dres, 0, rows, H, eps, stride6, 4 * H, modulated);

  // ---- backward: the attention half --------------------------------
  _ops.gated_idx_bwd(enc, dres, 0, mod, me, _aidx, 0, _df, 0, rows, H,
                     stride6, 2 * H, modulated);
  // d(attn out) = da W_out ([H, I])
  if (!back_gemm_(enc, b.out, H, I, _df, 0, _dob, 0, rows)) {
    return fail("an out projection is not readable");
  }
  lora_bwd_(enc, refiner, L, &BlockLora::out, a.ob, I, _df, H, _dob, I, rows,
            H, I);
  transpose(_dob, _doh, rows, NH);
  _ops.fill_f32(enc, _dk32, 0, 0.0f, (std::size_t)rows * I);
  _ops.fill_f32(enc, _dv32, 0, 0.0f, (std::size_t)rows * I);
  {
    train::AttnSegment sg;
    sg.q_row0 = 0; sg.qL = rows; sg.kL = rows; sg.qL_off = 0;
    sg.causal = false;
    _ops.attn_backward(enc, a.qh, a.kh, a.vh, a.oh, _doh, a.mlm, a.mls, _lse2,
                       _dsum, _dqh, _dk32, _dv32, NH, rows, HD, ascale, {sg});
  }
  // The fused [rows, 3I] gradient in the base's grouping, over the
  // normed copy (dead since the transposes).
  _ops.rope_half_bwd(enc, _dqh, false, _qkvn, 0, _rcos, _rsin, NH, rows, HD,
                     rot, 3 * I, Q_OFF, QKV_HS);
  _ops.rope_half_bwd(enc, _dk32, true, _qkvn, 0, _rcos, _rsin, NH, rows, HD,
                     rot, 3 * I, K_OFF, QKV_HS);
  _ops.rope_half_bwd(enc, _dv32, true, _qkvn, 0, _rcos, _rsin, NH, rows, HD,
                     0, 3 * I, V_OFF, QKV_HS);
  _ops.rms_heads_bwd(enc, a.qkvp, _qkvn, b.qn, rows, NH, HD, 3 * I, Q_OFF,
                     QKV_HS, c.qk_norm_eps);
  _ops.rms_heads_bwd(enc, a.qkvp, _qkvn, b.kn, rows, NH, HD, 3 * I, K_OFF,
                     QKV_HS, c.qk_norm_eps);
  // dh1 = dqkv W_qkv ([3I, H])
  if (!back_gemm_(enc, b.qkv, 3 * I, H, _qkvn, 0, _dh, 0, rows)) {
    return fail("a q|k|v projection is not readable");
  }
  lora_bwd_(enc, refiner, L, &BlockLora::qkv, a.h1, H, _qkvn, 3 * I, _dh, H,
            rows, 3 * I, H);
  _ops.rms_mod_bwd(enc, x, xe, _dh, 0, false, b.n1, mod, me, _aidx, 0,
                   dres, 0, rows, H, eps, stride6, H, modulated);
  // The meter (perf-flops): the attention's backward, five matmul units
  // against the forward's two -- 2.5x it, FlashAttention-2's count.
  m._flops.add(attention_flops(NH, (long long)rows * rows, HD) / 2 * 5);
  return true;
}

bool
MiniMaxH3Trainer::step(const Sample& smp, Result* out, std::string* err,
                       bool keep_pred)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "minimax-h3 trainer: " + m; }
    return false;
  };
  if (_p == nullptr || _slot < 0) { return fail("no adapter bound"); }
  if (smp.layout == nullptr || smp.text == nullptr) {
    return fail("an incomplete sample");
  }
  M& m = *_m;
  const M::Config& c = m._cfg;
  if (m.adaln_baked()) {
    return fail("the AdaLN is baked for a schedule; a training step's "
                "sigma is its own (load the model unbaked, or pruned)");
  }
  if (m.two_time_active()) {
    return fail("a flow-map adapter is live; set its strength to 0 for "
                "training steps");
  }
  const h3::PackedLayout& L = *smp.layout;
  const int seq = L.seq_len, n_text = L.num_text_rows;
  const int H = c.hidden, NL = c.n_layers;
  const int VPE = c.video_patch_elems(), AC = c.audio_channels;
  if (!inputs_(smp, err)) { return false; }
  const int n_t = (int)_uniq.size();
  if (!ensure_arena_(seq, n_text, n_t)) {
    return fail("arena allocation failed");
  }
  metal_compute::MetalCompute* mc = m._mc;
  auto commit = [&](CommandStream& st) {
    m._flops.flush(st);
    std::string ge;
    if (!st.commit().wait_ok(&ge)) { return fail(ge); }
    return true;
  };

  // ---- forward, keeping the checkpoints --------------------------------
  M::TrainTap tap;
  tap.ckpt = &_ckpt;
  tap.mods = &_mods;
  tap.fmod = &_fmod;
  tap.final_hidden = &_final;
  if (_train_refiner) {
    tap.refiner_in = &_ref_in;
    tap.refiner_out = &_ref_out;
  }
  // The rope tables, as the forward builds them: read by a kept block's
  // forward and by every backward.
  {
    const std::size_t tb = (std::size_t)seq * (c.rope_rot() / 2) * 4;
    if (_rcos.byte_size() < tb) { _rcos = mc->make_shared_buffer(tb); }
    if (_rsin.byte_size() < tb) { _rsin = mc->make_shared_buffer(tb); }
    if (_rcos.empty() || _rsin.empty()) { return fail("rope tables"); }
    m.build_rope_(L, _rcos, _rsin);
  }
  const std::size_t RH = (std::size_t)seq * H;
  const std::size_t MT = (std::size_t)n_t * c.adaln_out();
  // RECOMPUTE: NEVER. The forward runs each main block through the
  // trainer's own, keeping its activations for the backward below.
  std::string kerr;
  if (_keep_all) {
    tap.run_block = [this, MT, &kerr](ComputeEncoder& enc, int Lx,
                                      const Block& b, const SharedBuffer& x,
                                      int rows) {
      return block_pass_(enc, Pass::kForward, false, Lx, b, x, 0, rows,
                         _mods, (std::size_t)Lx * MT, _dres, keep_acts_(Lx),
                         &kerr);
    };
  }
  M::Step st;
  st.video = _xv.empty() ? nullptr : &_xv;
  st.audio = _xa.empty() ? nullptr : &_xa;
  st.text = smp.text;
  st.layout = &L;
  st.timesteps = &_uniq;
  st.row_timestep_index = &_row_idx;
  st.out_head = smp.out_head;
  st.dense_attention = true;
  st.train = &tap;
  const auto t_f0 = std::chrono::steady_clock::now();
  std::string ferr;
  M::Velocity v = m.forward(st, &ferr);
  const double fwd_ms = ms_since_(t_f0);
  if (v.empty()) {
    // A stop is not a failure: the forward's own stop check ("stopped")
    // reads here as the empty reason the stage takes for a stop.
    if (ferr.empty() || (_stop && _stop())) {
      if (err != nullptr) { err->clear(); }
      return false;
    }
    return fail(kerr.empty() ? ferr : kerr);
  }
  const auto t_b0 = std::chrono::steady_clock::now();

  // ---- the loss, and the heads' and final layer's backward -------------
  const int nvc = L.num_condition_video_rows, nvg = L.num_video_rows;
  const int nac = L.num_condition_audio_rows, nag = L.num_audio_rows;
  const bool audio_loss = nag > 0 && smp.audio_weight != 0.0f;
  const int n_v = nvg * VPE, n_a = nag * AC;
  double lv = 0.0, la = 0.0;
  {
    CommandStream cs = mc->make_command_stream();
    ComputeEncoder enc = cs.begin_compute();
    int gv = 0, ga = 0;
    // H3's target is v = x0 - noise: flow_loss forms pred - (noise - x0),
    // so the two are handed in swapped.
    if (n_v > 0) {
      const SharedBuffer pv =
          v.video.subview((std::size_t)nvc * VPE * 2, (std::size_t)n_v * 2);
      gv = _ops.flow_loss(enc, pv, *smp.video_noise, *smp.video_x0, _dpv,
                          _part, n_v, 2.0f * smp.weight / (float)n_v);
    }
    const std::size_t pa_off = (std::size_t)(gv + 1);
    if (audio_loss) {
      const SharedBuffer pa =
          v.audio.subview((std::size_t)nac * AC * 2, (std::size_t)n_a * 2);
      SharedBuffer part2 = _part.subview(
          pa_off * 4, _part.byte_size() - pa_off * 4);
      ga = _ops.flow_loss(enc, pa, *smp.audio_noise, *smp.audio_x0, _dpa,
                          part2, n_a,
                          2.0f * smp.weight * smp.audio_weight / (float)n_a);
    }
    // d(normed rows) = dpred W_head: only the generated rows' are not 0.
    _ops.fill_f32(enc, _dres, 0, 0.0f, (std::size_t)seq * H);
    const M::Linear& vout =
        smp.out_head >= 0 ? m._step_out[(std::size_t)smp.out_head]
                          : m._video_out;
    if (nvg > 0 &&
        !back_gemm_(enc, vout, VPE, H, _dpv, 0, _dproj,
                    (std::size_t)L.video_start * H, nvg)) {
      return fail("the video head is not readable");
    }
    if (audio_loss &&
        !back_gemm_(enc, m._audio_out, AC, H, _dpa, 0, _dproj,
                    (std::size_t)L.audio_start * H, nag)) {
      return fail("the audio head is not readable");
    }
    // The final layer's modulated norm, those rows only (fmod by timestep).
    if (nvg > 0) {
      const std::size_t o = (std::size_t)L.video_start * H;
      _ops.rms_mod_bwd(enc, _final, o, _dproj, o, false, m._final_norm, _fmod,
                       0, _tidx, (std::size_t)L.video_start, _dres, o, nvg,
                       H, c.final_norm_eps, 2 * H, H, true);
    }
    if (audio_loss) {
      const std::size_t o = (std::size_t)L.audio_start * H;
      _ops.rms_mod_bwd(enc, _final, o, _dproj, o, false, m._final_norm, _fmod,
                       0, _tidx, (std::size_t)L.audio_start, _dres, o, nag,
                       H, c.final_norm_eps, 2 * H, H, true);
    }
    enc.end();
    if (!commit(cs)) { return false; }
    const auto* pp = static_cast<const float*>(_part.contents());
    double sv = 0.0, sa = 0.0;
    for (int i = 0; i < gv; ++i) { sv += (double)pp[i]; }
    for (int i = 0; i < ga; ++i) { sa += (double)pp[pa_off + (std::size_t)i]; }
    lv = n_v > 0 ? sv / (double)n_v : 0.0;
    la = audio_loss ? sa / (double)n_a : 0.0;
  }

  // ---- the main blocks, in reverse -------------------------------------
  CommandStream bs = mc->make_command_stream();
  for (int Lx = NL - 1; Lx >= 0; --Lx) {
    if (_stop && _stop()) {
      if (_pf.valid()) { (void)_pf.get(); }
      if (err != nullptr) { err->clear(); }
      return false;
    }
    Block local;
    const Block* b = acquire_(Lx, local, err);
    if (b == nullptr) { return false; }
    const bool streamed = b != &m._blocks[(std::size_t)Lx];
    if (streamed && !m.fill_lora_(Lx)) {
      return fail(fmt("block {}'s adapter factors could not be read", Lx)());
    }
    {
      ComputeEncoder enc = bs.begin_compute();
      const bool ok = block_pass_(
          enc, _keep_all ? Pass::kBackward : Pass::kRecompute, false, Lx, *b,
          _ckpt, (std::size_t)Lx * RH, seq, _mods, (std::size_t)Lx * MT,
          _dres, _keep_all ? keep_acts_(Lx) : scratch_acts_(), err);
      enc.end();
      if (!ok) { return false; }
    }
    m._flops.flush(bs);
    CommandStream::Fence fence = bs.commit();
    // Streamed: the next block down is read under this one's GPU work,
    // into the slot this block is not using.
    if (streamed) { prefetch_(Lx - 1); }
    std::string ge;
    if (!fence.wait_ok(&ge)) {
      if (_pf.valid()) { (void)_pf.get(); }
      return fail(ge);
    }
  }

  // ---- the token refiner, over the text rows ---------------------------
  if (_train_refiner && c.n_refiner > 0) {
    {
      ComputeEncoder enc = bs.begin_compute();
      _ops.fill_f32(enc, _dres_t, 0, 0.0f, (std::size_t)n_text * H);
      // s.x's text rows = rms(refiner out) (+ a dial's constant shift).
      _ops.rms_mod_bwd(enc, _ref_out, 0, _dres, 0, true,
                       m._refiner_final_norm, _zero, 0, _aidx, 0, _dres_t, 0,
                       n_text, H, c.final_norm_eps, 0, 0, false);
      enc.end();
    }
    for (int R = c.n_refiner - 1; R >= 0; --R) {
      if (_stop && _stop()) {
        if (err != nullptr) { err->clear(); }
        return false;
      }
      {
        ComputeEncoder enc = bs.begin_compute();
        const bool ok = block_pass_(
            enc, Pass::kRecompute, true, R, m._refiner[(std::size_t)R],
            _ref_in, (std::size_t)R * n_text * H, n_text, _zero, 0, _dres_t,
            scratch_acts_(), err);
        enc.end();
        if (!ok) { return false; }
      }
      if (!commit(bs)) { return false; }
    }
  }
  if (m._fp8_failed) {
    m._fp8_failed = false;
    return fail("an FP8 weight could not be widened");
  }
  if (out != nullptr) {
    out->loss_video = lv;
    out->loss_audio = la;
    out->loss = smp.weight * (lv + (audio_loss ? smp.audio_weight * la : 0.0));
    out->forward_ms = fwd_ms;
    out->backward_ms = ms_since_(t_b0);
    if (keep_pred) {
      out->pred_video = std::move(v.video);
      out->pred_audio = std::move(v.audio);
    }
  }
  return true;
}

}  // namespace genai
}  // namespace vpipe
