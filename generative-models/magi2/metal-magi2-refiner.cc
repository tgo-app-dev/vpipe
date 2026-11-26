#include "generative-models/magi2/metal-magi2-refiner.h"

#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/magi2/magi2-accel.h"
#include "generative-models/magi2/metal-magi2-dit.h"
#include "generative-models/shared/metal-sage-attention.h"
#include "generative-models/shared/tp-cut.h"
#include "generative-models/shared/tp-overlap.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace vpipe {
namespace genai {

using metal_compute::ComputeEncoder;
using metal_compute::ComputeFunction;
using metal_compute::ComputeLibrary;
using metal_compute::FunctionConstants;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// The vendored steel attention's parameter blocks (steel/attn/params.h).
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
struct AttnMaskParams {
  std::int64_t M_strides[3];
};
struct AttnSpanParams {
  int video_start;
  int tokens_per_frame;
  int num_frames;
  int anchors;
  int qb_stride;
};

// gpu-kernels/metal/ops/magi2.metal's Magi2G32.
struct G32 {
  int M, N, K;
  int lda, ldb, ldc;
  long az, bz, cz;
  int has_bias, has_rs;
};

std::vector<int>
dim_blocks(int n, int b)
{
  std::vector<int> s;
  for (int i = 0; i < n; i += b) { s.push_back(std::min(b, n - i)); }
  return s;
}

// The bf16 a mask slot adds: 0, or a large
// negative, finite so no row's running maximum can become -inf.
std::uint16_t
mask_bits(bool keep)
{
  return keep ? 0 : 0xf149;   // -1.0e30 in bf16
}

}  // namespace

struct MetalMagi2Refiner::Kernels {
  ComputeLibrary lib, attn, nax;
  ComputeFunction g32a32, rms1f, inv_rmsf, qk_rope, gate, swiglu7,
      resid_add, round_f32, add;
  // [alignment]; Sage's [alignment][P*V in int8].
  ComputeFunction attn_fn[4], nax_fn[4], nax_sage[4][2];
};

// One layer's weights. The projections are one per modality (three in an
// mm layer); the per-head gate keeps the checkpoint's stacking.
struct MetalMagi2Refiner::Layer {
  int idx = 0, m = 1;
  SharedBuffer attn_norm, q_norm, k_norm, g;
  std::vector<magi2::Proj> qkv, proj, up, down;
  SharedBuffer mlp_norm;
};

struct MetalMagi2Refiner::Scratch {
  // The geometry the scratch and the sequence were built for.
  int T = -1, H = -1, W = -1, text_len = -1;
  Seq seq;
  std::vector<float> sin, cos;   // [n, R], in slot order
  // A holds a layer's normed rows, then its attention's output; qkv its
  // projections, then the attention's projection out.
  SharedBuffer S, A, qkv, g, U, act, D, sinb, cosb, in, mask, qb_off,
      qb_blocks, rs, vel;
  // A rank's band of normed rows for its MLP: A still holds the
  // attention's output of the bands not yet projected.
  SharedBuffer N;
};

MetalMagi2Refiner::~MetalMagi2Refiner()
{
  end_tp_generation();
}

// ---- tensor parallel --------------------------------------------------------

bool
MetalMagi2Refiner::share_of(const Config& c, Share* out, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "MAGI-2 refiner: tensor parallel: " + m; }
    return false;
  };
  Share z;
  if (!c.tp.active()) {
    z.h1 = c.n_q;
    z.kv1 = c.n_kv;
    z.f1 = c.mlp_inter;
    *out = z;
    return true;
  }
  const TpShare& t = c.tp;
  if (t.n_heads != c.n_q || t.n_ffn != c.mlp_inter) {
    return fail(fmt("the share counts {} heads and {} feed-forward rows "
                    "where this model has {} and {}", t.n_heads, t.n_ffn,
                    c.n_q, c.mlp_inter)());
  }
  const int g = c.n_q / std::max(1, c.n_kv);
  if (t.head0 < 0 || t.head1 > c.n_q || t.head0 >= t.head1 ||
      t.head0 % g != 0 || t.head1 % g != 0) {
    return fail(fmt("heads [{}, {}) of {} in whole groups of {}", t.head0,
                    t.head1, c.n_q, g)());
  }
  if (t.ffn0 < 0 || t.ffn1 > c.mlp_inter || t.ffn0 >= t.ffn1) {
    return fail(fmt("rows [{}, {}) of {}", t.ffn0, t.ffn1, c.mlp_inter)());
  }
  z.h0 = t.head0;
  z.h1 = t.head1;
  z.kv0 = t.head0 / g;
  z.kv1 = t.head1 / g;
  z.f0 = t.ffn0;
  z.f1 = t.ffn1;
  *out = z;
  return true;
}

TpShare
MetalMagi2Refiner::share_at(const Config& c, const TpShare& other)
{
  if (!other.active() || other.n_heads <= 0 || other.n_ffn <= 0) {
    return {};
  }
  TpShare t = other;
  t.n_heads = c.n_q;
  t.n_ffn = c.mlp_inter;
  // Each boundary scaled on its own (0 and the whole kept), the heads'
  // rounded to whole kv groups.
  const int g = c.n_q / std::max(1, c.n_kv);
  auto heads = [&](int h) {
    const double x = (double)h * c.n_q / other.n_heads;
    return std::clamp((int)std::lround(x / g) * g, 0, c.n_q);
  };
  auto rows = [&](int f) {
    return (int)((std::int64_t)f * c.mlp_inter / other.n_ffn);
  };
  t.head0 = heads(other.head0);
  t.head1 = heads(other.head1);
  t.ffn0 = rows(other.ffn0);
  t.ffn1 = rows(other.ffn1);
  return t;
}

bool
MetalMagi2Refiner::tp_buffer_(SharedBuffer& b, std::size_t bytes)
{
  if (!b.empty() && b.byte_size() >= bytes) { return true; }
  if (!b.empty() && _tp_pipe != nullptr) { _tp_pipe->forget(b); }
  b = _mc->make_shared_buffer(std::max<std::size_t>(bytes, 16));
  return !b.empty();
}

bool
MetalMagi2Refiner::begin_tp_generation(std::shared_ptr<TpChannel> ch,
                                       std::function<bool()> stop,
                                       std::uint64_t first_seq,
                                       std::string* err)
{
  auto fail = [&](const std::string& why) {
    if (err != nullptr) { *err = "tensor parallel: " + why; }
    return false;
  };
  end_tp_generation();
  if (!_cfg.tp.active()) { return fail("this refiner is not a rank"); }
  if (!ch) { return fail("no channel to the other ranks"); }
  auto pipe = std::make_unique<TpOverlap>(_mc, ch, kTpLanes, TpDType::kBF16);
  if (!pipe->valid()) {
    return fail("no events or thread for the exchanges");
  }
  _tp_ch = std::move(ch);
  _tp_pipe = std::move(pipe);
  _tp_seq = first_seq;
  try {
    _tp_ch->set_stop(std::move(stop));
  } catch (...) {
  }
  return true;
}

void
MetalMagi2Refiner::end_tp_generation()
{
  if (_tp_pipe == nullptr && !_tp_ch) { return; }
  if (_tp_pipe != nullptr && _tp_ch &&
      (_tp_pipe->pending() || _tp_pipe->failed())) {
    try {
      _tp_ch->set_stop([]() { return true; });
    } catch (...) {
    }
  }
  _tp_pipe.reset();
  if (_tp_ch) {
    for (int l = 0; l < kTpLanes; ++l) {
      for (const SharedBuffer* b : {&_tp_part[l], &_tp_recv[l]}) {
        if (b->empty()) { continue; }
        try {
          _tp_ch->forget(b->contents(), b->byte_size());
        } catch (...) {
        }
      }
    }
    try {
      _tp_ch->set_stop({});
    } catch (...) {
    }
  }
  _tp_ch.reset();
  _tp_seq = 0;
}

// ---- the sequence ----------------------------------------------------------

MetalMagi2Refiner::Seq
MetalMagi2Refiner::sequence(const Config& c, int T, int H, int W,
                            int text_len, int bq, int bk)
{
  Seq q;
  q.bq = bq;
  q.bk = bk;
  q.nv = T * H * W;
  const std::vector<int> ts = dim_blocks(T, c.block[0]);
  const std::vector<int> hs = dim_blocks(H, c.block[1]);
  const std::vector<int> ws = dim_blocks(W, c.block[2]);
  const int nt = (int)ts.size(), nh = (int)hs.size(), nw = (int)ws.size();
  const int nblk = nt * nh * nw;
  // Each block laid out contiguously, in (t, h, w) block order and (t, h,
  // w) order inside -- the release's _build_block_order -- and padded to
  // whole query tiles.
  std::vector<int> base((std::size_t)nblk + 1, 0);
  q.src.clear();
  q.src.reserve((std::size_t)q.nv + (std::size_t)nblk * bq + text_len);
  int t0 = 0;
  for (int bt = 0; bt < nt; ++bt) {
    int h0 = 0;
    for (int bh = 0; bh < nh; ++bh) {
      int w0 = 0;
      for (int bw = 0; bw < nw; ++bw) {
        const int b = (bt * nh + bh) * nw + bw;
        base[(std::size_t)b] = (int)q.src.size();
        for (int t = t0; t < t0 + ts[(std::size_t)bt]; ++t) {
          for (int h = h0; h < h0 + hs[(std::size_t)bh]; ++h) {
            for (int w = w0; w < w0 + ws[(std::size_t)bw]; ++w) {
              q.src.push_back((t * H + h) * W + w);
            }
          }
        }
        while (q.src.size() % (std::size_t)bq != 0) { q.src.push_back(-1); }
        w0 += ws[(std::size_t)bw];
      }
      h0 += hs[(std::size_t)bh];
    }
    t0 += ts[(std::size_t)bt];
  }
  q.nv_pad = (int)q.src.size();
  base[(std::size_t)nblk] = q.nv_pad;
  for (int j = 0; j < text_len; ++j) { q.src.push_back(q.nv + j); }
  q.n = (int)q.src.size();

  // The query tiles' key blocks. A video tile attends the blocks within
  // the radius in the block grid (ascending: block order is slot order)
  // and the text; a text tile attends everything.
  const int NQ = (q.n + bq - 1) / bq;
  const int NK = (q.n + bk - 1) / bk;
  const int text_kb0 = q.nv_pad / bk;
  q.qb_off.assign((std::size_t)NQ + 1, 0);
  q.qb_blocks.clear();
  int b = 0;
  for (int qt = 0; qt < NQ; ++qt) {
    const int row = qt * bq;
    if (row < q.nv_pad) {
      while (base[(std::size_t)b + 1] <= row) { ++b; }
      const int bt = b / (nh * nw), bh = (b / nw) % nh, bw = b % nw;
      // The window, clipped to the grid (the radius may be any size).
      const int t_lo = std::max(0, bt - c.radius[0]);
      const int t_hi = (int)std::min<long>(nt - 1, (long)bt + c.radius[0]);
      const int h_lo = std::max(0, bh - c.radius[1]);
      const int h_hi = (int)std::min<long>(nh - 1, (long)bh + c.radius[1]);
      const int w_lo = std::max(0, bw - c.radius[2]);
      const int w_hi = (int)std::min<long>(nw - 1, (long)bw + c.radius[2]);
      for (int kt = t_lo; kt <= t_hi; ++kt) {
        for (int kh = h_lo; kh <= h_hi; ++kh) {
          for (int kw = w_lo; kw <= w_hi; ++kw) {
            const int kb = (kt * nh + kh) * nw + kw;
            for (int k = base[(std::size_t)kb] / bk;
                 k < base[(std::size_t)kb + 1] / bk; ++k) {
              q.qb_blocks.push_back(k);
            }
          }
        }
      }
      for (int k = text_kb0; k < NK; ++k) { q.qb_blocks.push_back(k); }
    } else {
      for (int k = 0; k < NK; ++k) { q.qb_blocks.push_back(k); }
    }
    q.qb_off[(std::size_t)qt + 1] = (int)q.qb_blocks.size();
    long long keys = 0;
    for (int j = q.qb_off[(std::size_t)qt]; j < q.qb_off[(std::size_t)qt + 1];
         ++j) {
      keys += std::min(bk, q.n - q.qb_blocks[(std::size_t)j] * bk);
    }
    q.pairs += (long long)std::min(bq, q.n - row) * keys;
  }
  return q;
}

// ---- the configuration -----------------------------------------------------

bool
MetalMagi2Refiner::config_from_weights(WeightSet& ws, Config* out,
                                       std::string* err,
                                       const std::vector<int>& subset)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "MAGI-2 refiner: " + m; }
    return false;
  };
  const MetalLlamaWeights& src = ws.src();
  auto shape = [&](const std::string& n) -> std::vector<int64_t> {
    const auto* ti = src.info(n);
    return ti != nullptr ? ti->shape : std::vector<int64_t>{};
  };
  Config c;
  const auto ve = shape("pre_adapter.video_embedder.weight");
  const auto ae = shape("pre_adapter.audio_embedder.weight");
  const auto te = shape("pre_adapter.text_embedder.weight");
  if (ve.size() != 2 || ae.size() != 2 || te.size() != 2) {
    return fail("no pre_adapter embedders in " + ws.dir());
  }
  c.hidden = (int)ve[0];
  c.video_c = (int)ve[1];
  c.audio_c = (int)ae[1];
  c.text_c = (int)te[1];
  const auto bands = shape("pre_adapter.rope.bands");
  if (bands.size() != 1 || bands[0] * 8 != c.head_dim) {
    return fail("the rope's bands are not head_dim / 8");
  }
  if (ws.has("block.layers.0.mhc_norm.weight") ||
      ws.has("block.layers.0.attention.sinks")) {
    return fail("this is the preview DiT, not the refiner");
  }
  std::vector<int> all;
  for (int i = 0;; ++i) {
    if (!ws.has(fmt("block.layers.{}.attention.linear_qkv.weight", i)())) {
      break;
    }
    all.push_back(i);
  }
  if (all.empty()) { return fail("no block.layers"); }
  c.layers = subset.empty() ? all : subset;
  for (int li : c.layers) {
    const std::string p = fmt("block.layers.{}.", li)();
    const auto pn = shape(p + "attention.pre_norm.weight");
    const auto qkv = shape(p + "attention.linear_qkv.weight");
    const auto g = shape(p + "attention.linear_g.weight");
    const auto qn = shape(p + "attention.q_norm.weight");
    const auto dn = shape(p + "mlp.down_proj.weight");
    if (pn.size() != 1 || qkv.size() != 2 || g.size() != 2 ||
        qn.size() != 1 || dn.size() != 2) {
      return fail(fmt("layer {} is missing tensors", li)());
    }
    const int m = (int)(pn[0] / c.hidden);
    if ((m != 1 && m != 3) || pn[0] != (int64_t)m * c.hidden) {
      return fail(fmt("layer {}: the pre-norm's width", li)());
    }
    c.mm.push_back(m == 3 ? 1 : 0);
    if (qn[0] != (int64_t)m * c.head_dim) {
      return fail(fmt("layer {}: the q norm's width", li)());
    }
    const int nq = (int)(g[0] / m);
    const int rows = (int)(qkv[0] / m);
    const int nkv = (rows / c.head_dim - nq) / 2;
    if (nq <= 0 || nkv <= 0 || nq % nkv != 0 ||
        (nq + 2 * nkv) * c.head_dim != rows) {
      return fail(fmt("layer {}: the attention's heads", li)());
    }
    if (li == c.layers.front()) {
      c.n_q = nq;
      c.n_kv = nkv;
      c.mlp_inter = (int)dn[1];
    } else if (nq != c.n_q || nkv != c.n_kv || dn[1] != c.mlp_inter) {
      return fail(fmt("layer {}: a different shape", li)());
    }
  }
  *out = c;
  return true;
}

// ---- load ------------------------------------------------------------------

std::unique_ptr<MetalMagi2Refiner>
MetalMagi2Refiner::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                        const Config& cfg, std::string* err)
{
  std::unique_ptr<MetalMagi2Refiner> m(new MetalMagi2Refiner());
  m->_ws = std::move(ws);
  m->_mc = mc;
  m->_cfg = cfg;
  if (!m->load_(err)) { return nullptr; }
  return m;
}

bool
MetalMagi2Refiner::load_(std::string* err)
{
  auto fail = [&](const std::string& msg) {
    if (err != nullptr) { *err = "MAGI-2 refiner: " + msg; }
    return false;
  };
  if (_cfg.head_dim != 128) { return fail("head_dim is not 128"); }
  if (_cfg.mm.size() != _cfg.layers.size()) {
    return fail("the config's layers and mm flags differ");
  }
  {
    std::string se;
    if (!share_of(_cfg, &_sh, &se)) {
      if (err != nullptr) { *err = se; }
      return false;
    }
  }
  _k = std::make_unique<Kernels>();
  Kernels& k = *_k;
  k.lib = _mc->load_library("magi2_bf16");
  k.attn = _mc->load_library("attn_steel");
  struct Need {
    ComputeFunction* fn;
    const ComputeLibrary* lib;
    const char* name;
  };
  const Need need[] = {
      {&k.g32a32, &k.lib, "magi2_gemm32_a32"},
      {&k.rms1f, &k.lib, "magi2_rms1_f32"},
      {&k.inv_rmsf, &k.lib, "magi2_inv_rms_f32"},
      {&k.qk_rope, &k.lib, "magi2_qk_rope"},
      {&k.gate, &k.lib, "magi2_gate"},
      {&k.swiglu7, &k.lib, "magi2_swiglu7"},
      {&k.resid_add, &k.lib, "magi2_resid_add"},
      {&k.round_f32, &k.lib, "magi2_round_f32"},
      {&k.add, &k.lib, "magi2_add"},
  };
  for (const Need& n : need) {
    *n.fn = n.lib->function(n.name);
    if (!n.fn->valid()) {
      return fail(fmt("kernel '{}' did not load", n.name)());
    }
  }
  // The window: block-sparse spans (303) and one broadcast mask row
  // (300) for the padding keys. No sinks, not causal.
  auto window_fc = [](int a) {
    FunctionConstants fc;
    fc.set_bool(200, (a & 1) != 0).set_bool(201, (a & 2) != 0)
        .set_bool(300, true).set_bool(301, false).set_bool(302, false)
        .set_bool(303, true);
    return fc;
  };
  for (int a = 0; a < 4; ++a) {
    k.attn_fn[a] = k.attn.function("attn_steel_h_bd128_bf16", window_fc(a));
    if (!k.attn_fn[a].valid()) {
      return fail("the block-sparse steel attention did not load");
    }
  }
  // THE M5 ROUTES. Each takes over only when every entry it needs loads.
  auto env_on = [](const char* nm) {
    const char* e = std::getenv(nm);
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  };
  _g = std::make_unique<magi2::Gemms>();
  _g->set_flops(&_flops);
  {
    std::string ge;
    if (!_g->load(_mc, &ge)) { return fail(ge); }
  }
  _mma = _g->matrix_cores();
  if (_mc->supports_matrix_cores() && !env_on("VPIPE_MAGI2_NO_NAX_ATTN")) {
    k.nax = _mc->load_library("attn_steel_nax");
    bool all = k.nax.valid();
    for (int a = 0; a < 4 && all; ++a) {
      k.nax_fn[a] = k.nax.function("attn_steel_nax_h_bd128_bf16",
                                   window_fc(a));
      all = k.nax_fn[a].valid();
    }
    _nax = all;
  }
  // THE INT8 TIER, before the projections load: a holding tier converts
  // each one as it loads.
  if (_cfg.i8_tier >= 0) { _g->set_i8(_cfg.i8_tier, _cfg.autotune); }
  // SAGEATTENTION, on the matrix-core window attention alone.
  _sage_wanted =
      magi2::sage_runs(_cfg.sage, _mc->supports_matrix_cores()) && _nax;
  if (_sage_wanted && !load_sage_(err)) { return false; }
  if (_sage) {
    for (int a = 0; a < 4; ++a) {
      for (int pv = 0; pv < 2; ++pv) {
        FunctionConstants fc = window_fc(a);
        fc.set_bool(sage::kQkInt8Constant, true);
        MetalSageAttention::pv8_constants(fc, pv == 1);
        k.nax_sage[a][pv] =
            k.nax.function("attn_steel_nax_h_bd128_bf16", fc);
        if (pv == 1 && !_sage->pv8_for(_cfg.sage)) { continue; }
        if (!k.nax_sage[a][pv].valid()) {
          return fail("the int8 window attention did not load");
        }
      }
    }
  }

  WeightSet& ws = *_ws;
  const MetalLlamaWeights& src = ws.src();
  auto t = [&](const std::string& n) {
    return ws.tensor(n, _mc, WeightSet::Residency::Copied);
  };
  auto f32_host = [&](const std::string& n) -> std::vector<float> {
    SharedBuffer b = ws.read(n, _mc, WeightSet::Residency::Copied);
    const auto* ti = src.info(n);
    std::vector<float> v;
    if (b.empty() || ti == nullptr || ti->dtype != "F32") { return v; }
    v.resize(b.byte_size() / 4);
    std::memcpy(v.data(), b.contents(), v.size() * 4);
    return v;
  };
  auto dtype_is = [&](const std::string& n, const char* dt) {
    const auto* ti = src.info(n);
    return ti != nullptr && ti->dtype == dt;
  };

  _bands = f32_host("pre_adapter.rope.bands");
  if ((int)_bands.size() * 8 != _cfg.head_dim) {
    return fail("the rope's bands");
  }
  const char* names[2] = {"video", "text"};
  for (int i = 0; i < 2; ++i) {
    const std::string w = fmt("pre_adapter.{}_embedder.weight", names[i])();
    const std::string b = fmt("pre_adapter.{}_embedder.bias", names[i])();
    if (!dtype_is(w, "F32") || !dtype_is(b, "F32")) {
      return fail(fmt("the {} embedder is not f32", names[i])());
    }
    _emb_w[i] = t(w);
    _emb_b[i] = t(b);
  }
  {
    // The video head: RMSNorm (w + 1) folded into the f32 linear's
    // columns; the GEMM over the raw row times its inverse RMS is the
    // release's normalize-then-project.
    const int H = _cfg.hidden;
    const std::vector<float> nw =
        f32_host("post_adapter.final_norm_video.weight");
    std::vector<float> lw =
        f32_host("post_adapter.final_linear_video.weight");
    if ((int)nw.size() != H ||
        lw.size() != (std::size_t)_cfg.video_c * H) {
      return fail("the video head's shapes");
    }
    for (int r = 0; r < _cfg.video_c; ++r) {
      for (int ci = 0; ci < H; ++ci) {
        lw[(std::size_t)r * H + ci] *= nw[(std::size_t)ci] + 1.0f;
      }
    }
    _out_w = _mc->make_shared_buffer(lw.size() * 4);
    if (_out_w.empty()) { return fail("the video head"); }
    std::memcpy(_out_w.contents(), lw.data(), lw.size() * 4);
  }

  for (std::size_t li = 0; li < _cfg.layers.size(); ++li) {
    Layer L;
    L.idx = _cfg.layers[li];
    L.m = _cfg.mm[li] ? 3 : 1;
    const std::string p = fmt("block.layers.{}.", L.idx)();
    for (const char* f32n :
         {"attention.pre_norm.weight", "attention.q_norm.weight",
          "attention.k_norm.weight", "mlp.pre_norm.weight"}) {
      if (!dtype_is(p + f32n, "F32")) {
        return fail(fmt("layer {}: {} is not f32", L.idx, f32n)());
      }
    }
    for (const char* bfn :
         {"attention.linear_qkv.weight", "attention.linear_g.weight",
          "attention.linear_proj.weight", "mlp.up_gate_proj.weight",
          "mlp.down_proj.weight"}) {
      if (!dtype_is(p + bfn, "BF16")) {
        return fail(fmt("layer {}: {} is not bf16", L.idx, bfn)());
      }
    }
    const bool rank = _cfg.tp.active();
    const Share& sh = _sh;
    L.attn_norm = t(p + "attention.pre_norm.weight");
    L.q_norm = t(p + "attention.q_norm.weight");
    L.k_norm = t(p + "attention.k_norm.weight");
    if (!rank) {
      L.g = t(p + "attention.linear_g.weight");
    } else {
      // A rank's heads of each modality's gate rows.
      std::vector<std::pair<int, int>> gr;
      for (int i = 0; i < L.m; ++i) {
        gr.emplace_back(i * _cfg.n_q + sh.h0, i * _cfg.n_q + sh.h1);
      }
      L.g = tp_read_rows(_mc, src, p + "attention.linear_g.weight", "BF16",
                         L.m * _cfg.n_q, gr);
    }
    L.mlp_norm = t(p + "mlp.pre_norm.weight");
    for (const SharedBuffer* b :
         {&L.attn_norm, &L.q_norm, &L.k_norm, &L.g, &L.mlp_norm}) {
      if (b->empty()) {
        return fail(fmt("layer {}: a tensor did not read", L.idx)());
      }
    }
    const int H = _cfg.hidden, I = _cfg.mlp_inter, D = _cfg.head_dim;
    const int qs = _cfg.n_q * D, kvs = _cfg.n_kv * D;
    const int ldqkv = qs + 2 * kvs;
    // A rank's share: rows of an output cut (its heads' q | k | v, its
    // MLP rows of the interleaved up | gate), columns of an input cut.
    struct P {
      const char* name;
      int n, k;
      std::vector<std::pair<int, int>> rows;
      int c0, c1;
      std::vector<magi2::Proj>* out;
    };
    const P projs[] = {
        {"attention.linear_qkv.weight", ldqkv, H,
         {{sh.h0 * D, sh.h1 * D},
          {qs + sh.kv0 * D, qs + sh.kv1 * D},
          {qs + kvs + sh.kv0 * D, qs + kvs + sh.kv1 * D}},
         0, 0, &L.qkv},
        {"attention.linear_proj.weight", H, qs, {{0, H}}, sh.h0 * D,
         sh.h1 * D, &L.proj},
        {"mlp.up_gate_proj.weight", 2 * I, H, {{2 * sh.f0, 2 * sh.f1}}, 0,
         0, &L.up},
        {"mlp.down_proj.weight", H, I, {{0, H}}, sh.f0, sh.f1, &L.down}};
    for (const P& pj : projs) {
      std::string pe;
      const bool ok =
          rank ? _g->load_cut(ws, p + pj.name, L.m, pj.n, pj.k, pj.rows,
                              pj.c0, pj.c1, pj.out, &pe)
               : _g->load_stacked(ws, p + pj.name, L.m, pj.n, pj.k,
                                  "magi2-refiner/", pj.out, &pe);
      if (!ok) { return fail(pe); }
    }
    _layers.push_back(std::move(L));
  }
  _stream = _mc->make_command_stream();
  return true;
}

// ---- scratch ---------------------------------------------------------------

namespace {

struct ScratchSizes {
  std::size_t S, A, qkv, g, U, act, D, sin, cos, in, mask, qb_off,
      qb_blocks, rs, vel, N, tp;
  std::size_t total() const
  {
    return S + A + qkv + g + U + act + D + sin + cos + in + mask + qb_off +
           qb_blocks + rs + vel + N + tp;
  }
};

// A config's widths as its model runs them (the whole model's where the
// share does not fit: never loaded).
MetalMagi2Refiner::Share
widths_(const MetalMagi2Refiner::Config& c)
{
  MetalMagi2Refiner::Share w;
  if (!MetalMagi2Refiner::share_of(c, &w, nullptr)) {
    MetalMagi2Refiner::Config whole = c;
    whole.tp = TpShare{};
    MetalMagi2Refiner::share_of(whole, &w, nullptr);
  }
  return w;
}

ScratchSizes
scratch_sizes_(const MetalMagi2Refiner::Config& c,
               const MetalMagi2Refiner::Seq& q, int text_len)
{
  ScratchSizes z{};
  const MetalMagi2Refiner::Share w = widths_(c);
  // Whole 16 KB pages, as the device holds a buffer.
  auto b = [](std::size_t bytes) {
    return (std::max<std::size_t>(bytes, 16) + 16383) / 16384 * 16384;
  };
  const std::size_t N = (std::size_t)q.n, H = (std::size_t)c.hidden;
  const std::size_t qs = (std::size_t)w.heads() * c.head_dim;
  const std::size_t kvs = (std::size_t)w.kv() * c.head_dim;
  const std::size_t rows = (std::size_t)std::min(c.mlp_rows, q.n);
  const std::size_t I = (std::size_t)w.rows();
  const int R = 3 * c.head_dim / 8;
  z.S = b(N * H * 4);
  z.A = b(N * std::max(H, qs) * 2);
  z.qkv = b(N * std::max(qs + 2 * kvs, H) * 2);
  z.g = b(N * w.heads() * 2);
  z.U = b(rows * 2 * I * 2);
  z.act = b(rows * I * 2);
  z.D = b(rows * H * 2);
  z.sin = b(N * R * 4);
  z.cos = b(N * R * 4);
  z.in = b((std::size_t)q.nv_pad * c.video_c * 4 + 256 +
           (std::size_t)text_len * c.text_c * 4);
  z.mask = b(N * 2);
  z.qb_off = b(q.qb_off.size() * 4);
  z.qb_blocks = b(q.qb_blocks.size() * 4);
  z.rs = b((std::size_t)q.nv_pad * 4);
  z.vel = b((std::size_t)q.nv_pad * c.video_c * 4);
  // A rank: its band of normed rows, and the four exchange lanes' partials
  // and sums, a band each.
  if (c.tp.active()) {
    z.N = b(rows * H * 2);
    z.tp = 8 * b(rows * H * 2);
  }
  return z;
}

// The block projections the int8 tier runs, for a sequence of n slots: the
// attention's over it, the MLP's over a chunk.
std::vector<magi2::Gemms::Shape>
i8_shapes_(const MetalMagi2Refiner::Config& c, int n)
{
  const MetalMagi2Refiner::Share w = widths_(c);
  const int H = c.hidden, I = w.rows();
  const int qs = w.heads() * c.head_dim;
  const int ldqkv = qs + 2 * w.kv() * c.head_dim;
  const int chunk = std::min(c.mlp_rows, n);
  // (A rank's output projection runs a band at a time too.)
  return {{ldqkv, H, 0}, {H, qs, c.tp.active() ? chunk : 0},
          {2 * I, H, chunk}, {H, I, chunk}};
}

}  // namespace

std::size_t
MetalMagi2Refiner::scratch_bytes(const Config& c, int T, int H, int W,
                                 int text_len, bool matrix_cores)
{
  if (T <= 0 || H <= 0 || W <= 0) { return 0; }
  const Seq q = matrix_cores ? sequence(c, T, H, W, text_len, 64, 32)
                             : sequence(c, T, H, W, text_len, 32, 16);
  std::size_t t = scratch_sizes_(c, q, text_len).total();
  // The int8 context's scratch: the attention's projections over the
  // sequence, the MLP's over a chunk (reserved at the top of a forward).
  t += magi2::Gemms::i8_scratch_bytes(matrix_cores ? c.i8_tier : -1,
                                      c.autotune, q.n, i8_shapes_(c, q.n));
  // SageAttention's prologue: Q and K (and V) in int8 over the sequence.
  if (magi2::sage_runs(c.sage, matrix_cores)) {
    const Share w = widths_(c);
    t += MetalSageAttention::private_bytes(
        w.heads(), w.kv(), q.n, q.n, c.head_dim,
        MetalSageAttention::nax_query_block(),
        MetalSageAttention::nax_key_block(), 0, 0, c.sage.pv8);
  }
  return t;
}

bool
MetalMagi2Refiner::ensure_scratch_(const Seq& q, std::string* err)
{
  Scratch& s = *_s;
  const ScratchSizes z = scratch_sizes_(_cfg, q, s.text_len);
  auto mk = [&](SharedBuffer& buf, std::size_t bytes) {
    if (buf.byte_size() < bytes) {
      buf = SharedBuffer();
      buf = _mc->make_shared_buffer(bytes);
    }
    return !buf.empty();
  };
  if (!mk(s.S, z.S) || !mk(s.A, z.A) || !mk(s.qkv, z.qkv) ||
      !mk(s.g, z.g) || !mk(s.U, z.U) ||
      !mk(s.act, z.act) || !mk(s.D, z.D) || !mk(s.sinb, z.sin) ||
      !mk(s.cosb, z.cos) || !mk(s.in, z.in) || !mk(s.mask, z.mask) ||
      !mk(s.qb_off, z.qb_off) || !mk(s.qb_blocks, z.qb_blocks) ||
      !mk(s.rs, z.rs) || !mk(s.vel, z.vel) ||
      (_cfg.tp.active() && !mk(s.N, z.N))) {
    if (err != nullptr) { *err = "MAGI-2 refiner: scratch allocation"; }
    _s.reset();
    return false;
  }
  return true;
}

void
MetalMagi2Refiner::release_scratch()
{
  _s.reset();
  if (_g) { _g->release_scratch(); }
  // SageAttention's prologue scratch with it: the object is made again at
  // the next forward.
  _sage.reset();
}

bool
MetalMagi2Refiner::load_sage_(std::string* err)
{
  bool fatal = false;
  _sage = MetalSageAttention::load_for_model(_mc, /*bf16=*/true, _cfg.sage,
                                             "magi2 refiner", &fatal);
  if (fatal || !_sage) {
    if (err != nullptr) {
      *err = "MAGI-2 refiner: SageAttention's kernels did not load";
    }
    _sage_wanted = false;
    return !fatal;
  }
  _sage->reserve_pv8(_sage->pv8_for(_cfg.sage));
  return true;
}

int
MetalMagi2Refiner::i8_tier() const noexcept
{
  return _g ? _g->i8_tier() : -1;
}

long
MetalMagi2Refiner::i8_gemms() const noexcept
{
  return _g ? _g->i8_gemms() : 0;
}

long
MetalMagi2Refiner::dense_gemms() const noexcept
{
  return _g ? _g->dense_gemms() : 0;
}

std::size_t
MetalMagi2Refiner::weight_bytes(WeightSet& ws, const Config& c)
{
  const MetalLlamaWeights& src = ws.src();
  std::size_t t = 0;
  auto add = [&](const std::string& n) {
    const auto* ti = src.info(n);
    if (ti != nullptr) { t += (std::size_t)ti->nbytes; }
  };
  for (const char* nm : {"video", "text"}) {
    add(fmt("pre_adapter.{}_embedder.weight", nm)());
    add(fmt("pre_adapter.{}_embedder.bias", nm)());
  }
  add("post_adapter.final_linear_video.weight");
  // A rank's share (Share) of the gate and the projections.
  const Share w = widths_(c);
  const int H = c.hidden, I = w.rows();
  const int qs = w.heads() * c.head_dim;
  const int ldqkv = qs + 2 * w.kv() * c.head_dim;
  for (std::size_t li = 0; li < c.layers.size(); ++li) {
    const std::string p = fmt("block.layers.{}.", c.layers[li])();
    const int m = li < c.mm.size() && c.mm[li] ? 3 : 1;
    for (const char* n :
         {"attention.pre_norm.weight", "attention.q_norm.weight",
          "attention.k_norm.weight", "mlp.pre_norm.weight"}) {
      add(p + n);
    }
    t += (std::size_t)m * w.heads() * H * 2;
    // The projections at the layout the tier holds them in.
    t += (std::size_t)m * (magi2::proj_bytes(ldqkv, H, c.i8_tier) +
                           magi2::proj_bytes(H, qs, c.i8_tier) +
                           magi2::proj_bytes(2 * I, H, c.i8_tier) +
                           magi2::proj_bytes(H, I, c.i8_tier));
  }
  return t;
}

std::size_t
MetalMagi2Refiner::resident_bytes() const
{
  std::size_t t = 0;
  for (const Layer& ly : _layers) {
    for (const SharedBuffer* b :
         {&ly.attn_norm, &ly.q_norm, &ly.k_norm, &ly.g, &ly.mlp_norm}) {
      t += b->byte_size();
    }
    for (const auto* ps : {&ly.qkv, &ly.proj, &ly.up, &ly.down}) {
      for (const magi2::Proj& pj : *ps) { t += pj.bytes(); }
    }
  }
  for (int i = 0; i < 2; ++i) {
    t += _emb_w[i].byte_size() + _emb_b[i].byte_size();
  }
  return t + _out_w.byte_size();
}

// ---- forward ---------------------------------------------------------------

bool
MetalMagi2Refiner::forward(const Sample& smp, std::vector<float>* vel,
                           std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "MAGI-2 refiner: " + m; }
    return false;
  };
  const Config& c = _cfg;
  const Kernels& k = *_k;
  if (smp.T <= 0 || smp.H <= 0 || smp.W <= 0 || smp.video == nullptr) {
    return fail("an empty clip");
  }
  if (smp.text_len > 0 && smp.text == nullptr) { return fail("no text"); }
  if (!_s) { _s = std::make_unique<Scratch>(); }
  Scratch& s = *_s;
  const int R = 3 * c.head_dim / 8;
  if (s.T != smp.T || s.H != smp.H || s.W != smp.W ||
      s.text_len != smp.text_len) {
    s.T = smp.T; s.H = smp.H; s.W = smp.W; s.text_len = smp.text_len;
    s.seq = _nax ? sequence(c, smp.T, smp.H, smp.W, smp.text_len, 64, 32)
                 : sequence(c, smp.T, smp.H, smp.W, smp.text_len, 32, 16);
    if (!ensure_scratch_(s.seq, err)) { return false; }
    // The rope, as the preview lays it out (video raster, then text),
    // gathered into slot order; padding slots unrotated.
    MetalMagi2Dit::Sample ps;
    ps.T = smp.T; ps.H = smp.H; ps.W = smp.W;
    ps.text_len = smp.text_len;
    const MetalMagi2Dit::Layout L = MetalMagi2Dit::layout(ps, _bands);
    const Seq& q = s.seq;
    s.sin.assign((std::size_t)q.n * R, 0.0f);
    s.cos.assign((std::size_t)q.n * R, 1.0f);
    for (int i = 0; i < q.n; ++i) {
      const int r = q.src[(std::size_t)i];
      if (r < 0) { continue; }
      std::memcpy(&s.sin[(std::size_t)i * R], &L.sin[(std::size_t)r * R],
                  (std::size_t)R * 4);
      std::memcpy(&s.cos[(std::size_t)i * R], &L.cos[(std::size_t)r * R],
                  (std::size_t)R * 4);
    }
    std::memcpy(s.sinb.contents(), s.sin.data(), s.sin.size() * 4);
    std::memcpy(s.cosb.contents(), s.cos.data(), s.cos.size() * 4);
    auto* mk = (std::uint16_t*)s.mask.contents();
    for (int i = 0; i < q.n; ++i) {
      mk[i] = mask_bits(q.src[(std::size_t)i] >= 0);
    }
    std::memcpy(s.qb_off.contents(), q.qb_off.data(), q.qb_off.size() * 4);
    std::memcpy(s.qb_blocks.contents(), q.qb_blocks.data(),
                q.qb_blocks.size() * 4);
  }
  const Seq& q = s.seq;
  const int n = q.n;
  const int H = c.hidden;
  // This rank's widths (the whole model's when not split).
  const Share& sh = _sh;
  const bool rank = c.tp.active();
  if (rank && _tp_pipe == nullptr) {
    return fail("tensor parallel: no channel to the other ranks (a "
                "generation opens one: begin_tp_generation)");
  }
  const int nq = sh.heads(), nkv = sh.kv();
  const int qs = nq * c.head_dim, kvs = nkv * c.head_dim;
  const int ldqkv = qs + 2 * kvs;
  // Modality spans: video [0, nv_pad), audio none, text [nv_pad, n).
  const int span0[3] = {0, q.nv_pad, q.nv_pad};
  const int spann[3] = {q.nv_pad, 0, n - q.nv_pad};

  // The inputs in slot order: video rows (padding zero), then text.
  {
    float* in = (float*)s.in.contents();
    const int nv = q.nv;
    for (int i = 0; i < q.nv_pad; ++i) {
      float* row = in + (std::size_t)i * c.video_c;
      const int r = q.src[(std::size_t)i];
      if (r < 0) {
        std::memset(row, 0, (std::size_t)c.video_c * 4);
        continue;
      }
      for (int ch = 0; ch < c.video_c; ++ch) {
        row[ch] = smp.video[(std::size_t)ch * nv + r];
      }
    }
    if (smp.text_len > 0) {
      std::memcpy((char*)s.in.contents() +
                      (((std::size_t)q.nv_pad * c.video_c * 4 + 255) &
                       ~(std::size_t)255),
                  smp.text, (std::size_t)smp.text_len * c.text_c * 4);
    }
  }
  const std::size_t text_in_off =
      ((std::size_t)q.nv_pad * c.video_c * 4 + 255) & ~(std::size_t)255;

  // With no encoder open: the int8 tier's split-K tuning of the shapes the
  // last forward met, and SageAttention's scratch at this sequence (built
  // once while it stays put).
  magi2::Gemms& g = *_g;
  g.tune_pending();
  if (!g.reserve(n, i8_shapes_(c, n))) { return fail("the int8 scratch"); }
  if (_sage_wanted && !_sage && !load_sage_(err)) { return false; }
  const bool pv8_step = _sage != nullptr && _sage->pv8_for(c.sage) &&
                        magi2::sage_pv8_at(c.sage, smp.step);
  if (_sage) {
    std::string se;
    if (!_sage->reserve(nq, nkv, n, n, c.head_dim, q.bq, q.bk,
                        _sage->pv8_for(c.sage), &se)) {
      return fail("SageAttention: " + se);
    }
  }

  // THE METER: what an aborted forward counted and never flushed goes.
  _flops.begin();
  ComputeEncoder enc = _stream.begin_compute();
  auto g32 = [&](const SharedBuffer& A, std::size_t a_off, int lda,
                 const SharedBuffer& B, const SharedBuffer* bias,
                 const SharedBuffer* rs, const SharedBuffer& C,
                 std::size_t c_off, int ldc, int M, int N, int K) {
    if (M <= 0 || N <= 0) { return; }
    _flops.gemm(M, N, K);
    G32 p{M, N, K, lda, K, ldc, 0, 0, 0, bias != nullptr ? 1 : 0,
          rs != nullptr ? 1 : 0};
    enc.set_function(k.g32a32);
    enc.set_buffer(0, A, a_off);
    enc.set_buffer(1, B);
    enc.set_buffer(2, bias != nullptr ? *bias : B);
    enc.set_buffer(3, rs != nullptr ? *rs : B);
    enc.set_buffer(4, C, c_off);
    enc.set_constant_bytes(5, &p, sizeof p);
    enc.dispatch({(unsigned)((N + 31) / 32 * 128), (unsigned)((M + 31) / 32),
                  1},
                 {128, 1, 1});
  };
  // Over each modality span: fn(span index, first row, rows).
  auto spans = [&](int m, auto&& fn) {
    if (m == 1) {
      fn(0, 0, n);
      return;
    }
    for (int i = 0; i < 3; ++i) {
      if (spann[i] > 0) { fn(i, span0[i], spann[i]); }
    }
  };
  // A per-modality projection over each span's rows (m == 1: one over
  // all of them), through its route -- int8 where the tier takes it.
  auto mm_proj = [&](int m, const SharedBuffer& x,
                     const std::vector<magi2::Proj>& ps,
                     const SharedBuffer& y) {
    spans(m, [&](int i, int r0, int rn) {
      const magi2::Proj& pj = ps[(std::size_t)(m == 1 ? 0 : i)];
      g.proj(enc, x, (std::size_t)r0 * pj.k * 2, pj, y,
             (std::size_t)r0 * pj.n * 2, rn);
    });
  };
  auto rms_rows = [&](int m, const SharedBuffer& w, int r0, int rn,
                      int widx) {
    if (rn <= 0) { return; }
    enc.set_function(k.rms1f);
    enc.set_buffer(0, s.S, (std::size_t)r0 * H * 4);
    enc.set_buffer(1, s.A, (std::size_t)r0 * H * 2);
    enc.set_buffer(2, w, (std::size_t)(m == 1 ? 0 : widx) * H * 4);
    enc.set_constant(3, H);
    enc.set_constant(4, c.eps);
    enc.dispatch({(unsigned)(256 * rn), 1, 1}, {256, 1, 1});
  };
  auto resid = [&](const SharedBuffer& y, std::size_t y_off, int ldy,
                   int r0, int rn, bool rnd) {
    enc.set_function(k.resid_add);
    enc.set_buffer(0, s.S, (std::size_t)r0 * H * 4);
    enc.set_buffer(1, y, y_off);
    enc.set_constant(2, H);
    enc.set_constant(3, ldy);
    enc.set_constant(4, rnd ? 1 : 0);
    enc.dispatch({(unsigned)H, (unsigned)rn, 1}, {256, 1, 1});
  };

  // ---- embedding into the f32 stream, rounded to params_dtype ----------
  g32(s.in, 0, c.video_c, _emb_w[0], &_emb_b[0], nullptr, s.S, 0, H,
      q.nv_pad, H, c.video_c);
  g32(s.in, text_in_off, c.text_c, _emb_w[1], &_emb_b[1], nullptr, s.S,
      (std::size_t)q.nv_pad * H * 4, H, n - q.nv_pad, H, c.text_c);
  enc.set_function(k.round_f32);
  enc.set_buffer(0, s.S);
  enc.set_constant(1, H);
  enc.dispatch({(unsigned)H, (unsigned)n, 1}, {256, 1, 1});

  // ---- the attention's fixed parameters ------------------------------
  SteelAttnParams pp{};
  {
    constexpr int HD = 128;
    pp.B = 1; pp.H = nq; pp.D = HD;
    pp.qL = n; pp.kL = n;
    pp.gqa_factor = nq / nkv;
    pp.scale = 1.0f / std::sqrt((float)HD);
    pp.NQ = (n + q.bq - 1) / q.bq;
    pp.NK = (n + q.bk - 1) / q.bk;
    pp.NQ_aligned = n / q.bq;
    pp.NK_aligned = n / q.bk;
    pp.qL_rem = n - pp.NQ_aligned * q.bq;
    pp.kL_rem = n - pp.NK_aligned * q.bk;
    pp.qL_off = 0;
    for (std::int64_t* st : {pp.Q_strides, pp.K_strides, pp.V_strides}) {
      st[0] = (std::int64_t)n * ldqkv; st[1] = HD; st[2] = ldqkv;
    }
    // O into A, whose rows the q|k|v and gate GEMMs have read.
    pp.O_strides[0] = (std::int64_t)n * qs;
    pp.O_strides[1] = HD;
    pp.O_strides[2] = qs;
  }
  const AttnMaskParams mp{{0, 0, 0}};
  const AttnSpanParams sp{0, 0, 0, 0, 0};
  const int al = (n % q.bq == 0 ? 1 : 0) | (n % q.bk == 0 ? 2 : 0);
  const ComputeFunction& attn = _nax ? k.nax_fn[al] : k.attn_fn[al];
  const int bound[2] = {0, 0};

  // ---- a RANK's bands (Config::tp) -------------------------------------
  // The output projection and the MLP in row BANDS -- each modality
  // span's rows in chunks of mlp_rows -- each yielding a PARTIAL sum the
  // other ranks complete. A handoff is encoded between encoders into the
  // open command buffer: the lane's signal after the dispatch that wrote
  // its partial, a wait before the first that reads the sum (TpOverlap).
  // OVERLAPPED, GPU step i runs A(i) -- band i's projection, posted --
  // then B(i - 1) -- that band's sum added, its MLP, posted -- then
  // C(i - 2) -- its sum added: band i's first exchange crosses under
  // A(i + 1), its second under A(i + 2) and B(i + 1); the last band's MLP
  // alone is exposed, once a layer. VPIPE_MAGI2_TP_LAG=0 runs A, B, C a
  // band at a time: the same arithmetic.
  const bool lag = rank && [] {
    const char* e = std::getenv("VPIPE_MAGI2_TP_LAG");
    return !(e != nullptr && std::atoi(e) == 0);
  }();
  _tp_lagged = lag;
  const int band_rows = std::min(c.mlp_rows, n);
  if (rank) {
    const std::size_t pb = (std::size_t)band_rows * (std::size_t)H * 2;
    for (int l = 0; l < kTpLanes; ++l) {
      if (!tp_buffer_(_tp_part[l], pb) || !tp_buffer_(_tp_recv[l], pb)) {
        enc.end();
        _stream.commit().wait();
        return fail("tensor parallel: no room for the partial sums");
      }
    }
  }
  auto tp_post = [&](int lane, int rows) {
    const TpOverlap::Ticket t =
        _tp_pipe->post(lane, _tp_part[lane], _tp_recv[lane],
                       (std::size_t)rows * (std::size_t)H * 2, _tp_seq++);
    ++_tp_exchanges;
    enc.end();
    _stream.encode_signal(*t.ready, t.ready_at);
    enc = _stream.begin_compute();
    return t.done_at;
  };
  // The others' sum awaited and added to this rank's, into the lane's
  // sum buffer: both ranks add the same two, so their streams agree.
  auto tp_sum = [&](int lane, std::uint64_t done_at, int rows) {
    enc.end();
    _stream.encode_wait(_tp_pipe->done(), done_at);
    enc = _stream.begin_compute();
    enc.set_function(k.add);
    enc.set_buffer(0, _tp_part[lane]);
    enc.set_buffer(1, _tp_recv[lane]);
    enc.set_buffer(2, _tp_recv[lane]);
    const unsigned cnt = (unsigned)((std::size_t)rows * H);
    enc.set_constant(3, cnt);
    enc.dispatch({cnt, 1, 1}, {256, 1, 1});
  };
  auto tp_bands = [&](const Layer& ly, bool first) {
    struct Band {
      int w, c0, cn;
    };
    std::vector<Band> bands;
    spans(ly.m, [&](int i, int r0, int rn) {
      for (int c0 = r0; c0 < r0 + rn; c0 += c.mlp_rows) {
        bands.push_back({ly.m == 1 ? 0 : i, c0,
                         std::min(c.mlp_rows, r0 + rn - c0)});
      }
    });
    const int nbn = (int)bands.size();
    std::vector<std::uint64_t> va((std::size_t)nbn), vb((std::size_t)nbn);
    const int I = sh.rows();
    auto step_a = [&](int b) {
      const Band& bd = bands[(std::size_t)b];
      const magi2::Proj& pj = ly.proj[(std::size_t)bd.w];
      g.proj(enc, s.A, (std::size_t)bd.c0 * pj.k * 2, pj, _tp_part[b & 1],
             0, bd.cn);
      va[(std::size_t)b] = tp_post(b & 1, bd.cn);
    };
    auto step_b = [&](int b) {
      const Band& bd = bands[(std::size_t)b];
      tp_sum(b & 1, va[(std::size_t)b], bd.cn);
      resid(_tp_recv[b & 1], 0, H, bd.c0, bd.cn, first);
      // The band's normed rows into N: A still holds the attention's
      // output of the bands to come.
      enc.set_function(k.rms1f);
      enc.set_buffer(0, s.S, (std::size_t)bd.c0 * H * 4);
      enc.set_buffer(1, s.N);
      enc.set_buffer(2, ly.mlp_norm, (std::size_t)bd.w * H * 4);
      enc.set_constant(3, H);
      enc.set_constant(4, c.eps);
      enc.dispatch({(unsigned)(256 * bd.cn), 1, 1}, {256, 1, 1});
      g.proj(enc, s.N, 0, ly.up[(std::size_t)bd.w], s.U, 0, bd.cn);
      enc.set_function(k.swiglu7);
      enc.set_buffer(0, s.U);
      enc.set_buffer(1, s.act);
      enc.set_constant(2, I);
      enc.dispatch({(unsigned)I, (unsigned)bd.cn, 1}, {64, 1, 1});
      g.proj(enc, s.act, 0, ly.down[(std::size_t)bd.w], _tp_part[2 + (b & 1)],
             0, bd.cn);
      vb[(std::size_t)b] = tp_post(2 + (b & 1), bd.cn);
    };
    auto step_c = [&](int b) {
      const Band& bd = bands[(std::size_t)b];
      tp_sum(2 + (b & 1), vb[(std::size_t)b], bd.cn);
      resid(_tp_recv[2 + (b & 1)], 0, H, bd.c0, bd.cn, false);
    };
    if (!lag) {
      for (int b = 0; b < nbn; ++b) {
        step_a(b);
        step_b(b);
        step_c(b);
      }
      return;
    }
    for (int i = 0; i < nbn + 2; ++i) {
      if (i < nbn) { step_a(i); }
      if (i >= 1 && i - 1 < nbn) { step_b(i - 1); }
      if (i >= 2 && i - 2 < nbn) { step_c(i - 2); }
    }
  };

  // ---- the layers ------------------------------------------------------
  int pos = 0;
  for (const Layer& ly : _layers) {
    const bool first = pos == 0;
    // Attention.
    spans(ly.m, [&](int i, int r0, int rn) {
      rms_rows(ly.m, ly.attn_norm, r0, rn, i);
    });
    mm_proj(ly.m, s.A, ly.qkv, s.qkv);
    // The per-head gate stays bf16: 32 outputs a row.
    spans(ly.m, [&](int i, int r0, int rn) {
      g.dense(enc, s.A, (std::size_t)r0 * H * 2, ly.g,
              (std::size_t)(ly.m == 1 ? 0 : i) * nq * H * 2, s.g,
              (std::size_t)r0 * nq * 2, rn, nq, H);
    });
    spans(ly.m, [&](int i, int r0, int rn) {
      const std::size_t wo =
          (std::size_t)(ly.m == 1 ? 0 : i) * c.head_dim * 4;
      enc.set_function(k.qk_rope);
      enc.set_buffer(0, s.qkv);
      enc.set_buffer(1, s.sinb);
      enc.set_buffer(2, s.cosb);
      enc.set_buffer(3, ly.q_norm, wo);
      enc.set_buffer(4, ly.k_norm, wo);
      enc.set_constant(5, ldqkv);
      enc.set_constant(6, nq);
      enc.set_constant(7, nkv);
      enc.set_constant(8, R);
      enc.set_constant(9, c.eps);
      enc.set_constant(10, r0);
      enc.dispatch({(unsigned)(32 * (nq + nkv)), (unsigned)rn, 1},
                   {32, 1, 1});
    });
    // SageAttention: Q and K (and V) quantized into its scratch, then the
    // window attention's int8 twin over them.
    const bool sage = _sage != nullptr && pos >= c.sage.dense_layers;
    const bool pv8 = sage && pv8_step;
    if (sage) {
      using Op = MetalSageAttention::Operand;
      const Op qo{&s.qkv, 0, ldqkv, c.head_dim};
      const Op ko{&s.qkv, (std::size_t)qs, ldqkv, c.head_dim};
      const Op vo{&s.qkv, (std::size_t)(qs + kvs), ldqkv, c.head_dim};
      sage::Config sc = c.sage;
      sc.pv8 = pv8;
      std::string se;
      if (!_sage->prepare(enc, qo, ko, nq, nkv, n, n, c.head_dim,
                          q.bq, q.bk, sc, &se, pv8 ? &vo : nullptr)) {
        enc.end();
        _stream.commit().wait();
        return fail("SageAttention: " + se);
      }
      ++_n_sage;
      if (pv8) { ++_n_sage_pv8; }
    }
    enc.set_function(sage ? k.nax_sage[al][pv8 ? 1 : 0] : attn);
    enc.set_buffer(0, s.qkv);
    enc.set_buffer(1, s.qkv, (std::size_t)qs * 2);
    enc.set_buffer(2, s.qkv, (std::size_t)(qs + kvs) * 2);
    enc.set_buffer(3, s.A);
    enc.set_constant_bytes(4, &pp, sizeof pp);
    enc.set_constant_bytes(5, &mp, sizeof mp);
    enc.set_buffer(6, s.mask);
    enc.set_buffer(8, s.qb_off);
    enc.set_buffer(9, s.qb_blocks);
    enc.set_constant_bytes(10, &sp, sizeof sp);
    enc.set_constant_bytes(11, bound, sizeof bound);
    if (sage) { _sage->bind(enc); }
    enc.dispatch({32u * (unsigned)pp.NQ, 4u * (unsigned)nq, 1},
                 {32, 4, 1});
    _flops.attention(nq, q.pairs, c.head_dim);
    enc.set_function(k.gate);
    enc.set_buffer(0, s.A);
    enc.set_buffer(1, s.g);
    enc.set_constant(2, nq);
    enc.set_constant(3, c.head_dim);
    enc.set_constant(4, nq);
    enc.set_constant(5, 0);
    enc.dispatch({(unsigned)qs, (unsigned)n, 1}, {128, 1, 1});
    if (rank) {
      tp_bands(ly, first);
      ++pos;
      enc.end();
      _flops.flush(_stream);
      std::string ge;
      if (!_stream.commit().wait_ok(&ge)) {
        return fail(fmt("layer {}: {}", ly.idx, ge)());
      }
      if (_tp_pipe->failed()) {
        return fail("tensor parallel: " + _tp_pipe->error());
      }
      if (_block_cb && !_block_cb(pos, (int)_layers.size())) {
        if (err != nullptr) { err->clear(); }
        return false;
      }
      enc = _stream.begin_compute();
      continue;
    }
    // The projection into qkv's buffer (free again), added into the
    // stream: the first layer's attention add is the release's bf16 +
    // bf16.
    mm_proj(ly.m, s.A, ly.proj, s.qkv);
    resid(s.qkv, 0, H, 0, n, first);

    // MLP, in row chunks within each modality span.
    spans(ly.m, [&](int i, int r0, int rn) {
      rms_rows(ly.m, ly.mlp_norm, r0, rn, i);
    });
    const int I = c.mlp_inter;
    spans(ly.m, [&](int i, int r0, int rn) {
      const std::size_t wsel = (std::size_t)(ly.m == 1 ? 0 : i);
      for (int c0 = r0; c0 < r0 + rn; c0 += c.mlp_rows) {
        const int cn = std::min(c.mlp_rows, r0 + rn - c0);
        g.proj(enc, s.A, (std::size_t)c0 * H * 2, ly.up[wsel], s.U, 0, cn);
        enc.set_function(k.swiglu7);
        enc.set_buffer(0, s.U);
        enc.set_buffer(1, s.act);
        enc.set_constant(2, I);
        enc.dispatch({(unsigned)I, (unsigned)cn, 1}, {64, 1, 1});
        g.proj(enc, s.act, 0, ly.down[wsel], s.D, 0, cn);
        resid(s.D, 0, H, c0, cn, false);
      }
    });
    ++pos;
    // A command buffer a layer: the progress callback and a stop between.
    enc.end();
    _flops.flush(_stream);
    std::string ge;
    if (!_stream.commit().wait_ok(&ge)) {
      return fail(fmt("layer {}: {}", ly.idx, ge)());
    }
    if (_block_cb && !_block_cb(pos, (int)_layers.size())) {
      if (err != nullptr) { err->clear(); }
      return false;
    }
    enc = _stream.begin_compute();
  }

  // ---- the video head --------------------------------------------------
  enc.set_function(k.inv_rmsf);
  enc.set_buffer(0, s.S);
  enc.set_buffer(1, s.rs);
  enc.set_constant(2, H);
  enc.set_constant(3, c.eps);
  enc.dispatch({(unsigned)(256 * q.nv_pad), 1, 1}, {256, 1, 1});
  g32(s.S, 0, H, _out_w, nullptr, &s.rs, s.vel, 0, c.video_c, q.nv_pad,
      c.video_c, H);
  enc.end();
  _flops.flush(_stream);
  std::string ge;
  if (!_stream.commit().wait_ok(&ge)) { return fail("the head: " + ge); }
  const float* v = (const float*)s.vel.contents();
  vel->assign((std::size_t)c.video_c * q.nv, 0.0f);
  for (int i = 0; i < q.nv_pad; ++i) {
    const int r = q.src[(std::size_t)i];
    if (r < 0) { continue; }
    for (int ch = 0; ch < c.video_c; ++ch) {
      (*vel)[(std::size_t)ch * q.nv + r] = v[(std::size_t)i * c.video_c + ch];
    }
  }
  return true;
}

}  // namespace genai
}  // namespace vpipe
