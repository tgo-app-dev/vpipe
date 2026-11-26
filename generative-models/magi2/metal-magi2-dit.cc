#include "generative-models/magi2/metal-magi2-dit.h"

#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/magi2/magi2-accel.h"
#include "generative-models/shared/metal-sage-attention.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/shared/tp-cut.h"
#include "generative-models/shared/tp-overlap.h"
#include "generative-models/shared/wired-pool.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <future>

namespace vpipe {
namespace genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::ComputeFunction;
using metal_compute::ComputeLibrary;
using metal_compute::FunctionConstants;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// The steel / NAX flash kernels' AttnParams (vendored MLX).
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

// gpu-kernels/metal/ops/magi2.metal's Magi2G32.
struct G32 {
  int M, N, K;
  int lda, ldb, ldc;
  long az, bz, cz;
  int has_bias, has_rs;
};

std::uint16_t
to_bf16(float v)
{
  std::uint32_t u;
  std::memcpy(&u, &v, 4);
  if ((u & 0x7f800000u) == 0x7f800000u) { return (std::uint16_t)(u >> 16); }
  u += 0x7fffu + ((u >> 16) & 1u);
  return (std::uint16_t)(u >> 16);
}

constexpr int kMaxm = 32;   // the expert kernel's tile (kMagi2Tm)

}  // namespace

struct MetalMagi2Dit::Kernels {
  ComputeLibrary lib, elt, attn, nax, moe_nax;
  ComputeFunction g32a16, g32a32, embed_finish, mhc_pre, mhc_post, rms1,
      inv_rms, qk_rope, gate, swiglu7, topk, invert, expert, combine, add,
      drop_bin;
  ComputeFunction ifill, hist, sort_setup, scatter;
  // The routed experts on the matrix cores, over 64-row work items; and
  // without them, the staged simdgroup kernel.
  ComputeFunction expert_nax, tile_items, expert_tg;
  // [alignment]: the steel kernel, the matrix-core one, and Sage's
  // [alignment][P*V in int8] -- all with the per-head sinks.
  ComputeFunction attn_fn[4], nax_fn[4], nax_sage[4][2];
};

// One layer's weights. Per-modality tensors keep the checkpoint's stacking
// ([m * N, K]); `m` is 3 in an mm layer, else 1.
struct MetalMagi2Dit::Layer {
  int idx = 0, m = 1;
  bool moe = false;
  bool streamed = false;   // its experts read per forward into a slot
  SharedBuffer attn_norm, q_norm, k_norm, g, sinks;
  SharedBuffer mlp_norm;
  // The block projections, one per modality where the layer keeps three.
  std::vector<magi2::Proj> qkv, proj;
  std::vector<magi2::Proj> up, down;             // dense
  std::vector<magi2::Proj> split, merge, sh1, sh2, ms1, ms2;
  SharedBuffer rgate, rbias;
  SharedBuffer wx[3];                            // W_gate W_up W_down
  // The hyper-connections: phi folded with the norm weight per modality
  // and transposed, [m][24, n H] f32, and alpha | b_pre | b_post | b_res.
  SharedBuffer phi_attn, phi_mlp, ab_attn, ab_mlp;
  int mhc_m = 1;
};

struct MetalMagi2Dit::Scratch {
  int n = 0, chunk = 0;
  SharedBuffer S, D, coef, Y, A, qkv, g, O, P, U, act;
  SharedBuffer sin, cos, emb, in;
  SharedBuffer split, logits, eid, prob, hist, boff, tile2e, cursor, nt,
      srow, sdst, pos, yexp, comb, merge, x1, x2, a1, a2, y1, y2, items;
  SharedBuffer rs, vel;
  // A batch's samples after the first: their residual streams and rope
  // tables (the first uses S / sin / cos), kept for the next call.
  std::vector<SharedBuffer> bS, bsin, bcos, bcoef;
};

MetalMagi2Dit::~MetalMagi2Dit()
{
  // A rank's channel let go first: its worker joined, its buffers
  // forgotten, before they go with the model.
  end_tp_generation();
  // The held experts back out of the wired pool: only unwiring gives the
  // pool its bytes back, so a buffer dropped wired would leak them for the
  // rest of the run.
  if (_wire && _wire->on()) {
    for (Layer& ly : _layers) {
      if (!ly.moe || ly.streamed) { continue; }
      SharedBuffer* bufs[3] = {&ly.wx[0], &ly.wx[1], &ly.wx[2]};
      _wire->note_unwired(_wire->wire_set(bufs, false));
    }
  }
}

// ---- the sequence -----------------------------------------------------------

std::vector<float>
MetalMagi2Dit::time_embedding(float t, int dim)
{
  const int half = dim / 2;
  const float pos = t * 1000.0f;
  const float nl = (float)(-std::log(10000.0));
  std::vector<float> out((std::size_t)dim, 0.0f);
  for (int j = 0; j < half; ++j) {
    const float f = std::exp(nl * (float)j / (float)half);
    const float a = pos * f;
    out[(std::size_t)j] = std::cos(a);
    out[(std::size_t)(half + j)] = std::sin(a);
  }
  return out;
}

MetalMagi2Dit::Layout
MetalMagi2Dit::layout(const Sample& s, const std::vector<float>& bands)
{
  Layout L;
  const int B = (int)bands.size();
  const int R = 3 * B;
  int nimg = 0;
  for (const Image& im : s.images) { nimg += im.h * im.w; }
  const int nvg = s.T * s.H * s.W;
  L.nv = nvg + nimg;
  L.na = s.audio_len;
  L.nt = s.text_len + (int)s.images.size();
  L.n = L.nv + L.na + L.nt;
  L.coords.assign((std::size_t)L.n * 9, 0.0f);
  int r = 0;
  auto put = [&](float t, float h, float w, float T, float H, float W,
                 float rt, float rh, float rw) {
    float* c = &L.coords[(std::size_t)r * 9];
    c[0] = t; c[1] = h; c[2] = w; c[3] = T; c[4] = H; c[5] = W;
    c[6] = rt; c[7] = rh; c[8] = rw;
    ++r;
  };
  for (int t = 0; t < s.T; ++t) {
    for (int h = 0; h < s.H; ++h) {
      for (int w = 0; w < s.W; ++w) {
        put((float)t, (float)h, (float)w, (float)s.T, (float)s.H,
            (float)s.W, (float)s.T, (float)s.H, (float)s.W);
      }
    }
  }
  // Images sit past the clip on the time axis, one latent step apart.
  for (std::size_t k = 0; k < s.images.size(); ++k) {
    const Image& im = s.images[k];
    const float to = (float)(s.T + 2 + (int)k);
    for (int h = 0; h < im.h; ++h) {
      for (int w = 0; w < im.w; ++w) {
        put(to + 0.0f, (float)h, (float)w, 1.0f, (float)im.h, (float)im.w,
            1.0f, (float)im.h, (float)im.w);
      }
    }
  }
  const int La = s.audio_len;
  const float ref_t = (float)((La - 1) / 8 + 1);
  for (int i = 0; i < La; ++i) {
    put((float)i, 0, 0, (float)La, 1, 1, ref_t, 1, 1);
  }
  const int Lt = s.text_len;
  for (int i = 0; i < Lt; ++i) {
    put((float)(i - Lt), 0, 0, (float)Lt, 1, 1, 1, 1, 1);
  }
  for (std::size_t k = 0; k < s.images.size(); ++k) {
    const Image& im = s.images[k];
    const float to = (float)(s.T + 2 + (int)k);
    put(to, -1, -1, 1, (float)im.h, (float)im.w, 1, (float)im.h,
        (float)im.w);
  }
  L.sin.assign((std::size_t)L.n * R, 0.0f);
  L.cos.assign((std::size_t)L.n * R, 0.0f);
  for (int i = 0; i < L.n; ++i) {
    const float* c = &L.coords[(std::size_t)i * 9];
    for (int a = 0; a < 3; ++a) {
      const float size = c[3 + a], ref = c[6 + a];
      float scale = (ref - 1.0f) / (size - 1.0f);
      if (ref == 1.0f && size == 1.0f) { scale = 1.0f; }
      const float centre = a == 0 ? 0.0f : (size - 1.0f) / 2.0f;
      const float x = c[a] - centre;
      for (int b = 0; b < B; ++b) {
        const float p = x * scale * bands[(std::size_t)b];
        L.sin[(std::size_t)i * R + a * B + b] = std::sin(p);
        L.cos[(std::size_t)i * R + a * B + b] = std::cos(p);
      }
    }
  }
  return L;
}

// ---- the configuration ------------------------------------------------------

bool
MetalMagi2Dit::config_from_weights(WeightSet& ws, Config* out,
                                   std::string* err,
                                   const std::vector<int>& subset)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "MAGI-2 DiT: " + m; }
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
  c.hidden = (int)(ve[0] / c.streams);
  c.video_c = (int)ve[1];
  c.audio_c = (int)ae[1];
  c.text_c = (int)te[1];
  const auto bands = shape("pre_adapter.rope.bands");
  if (bands.size() != 1 || bands[0] * 8 != c.head_dim) {
    return fail("the rope's bands are not head_dim / 8");
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
    if (pn.size() != 1 || qkv.size() != 2 || g.size() != 2) {
      return fail(fmt("layer {} is missing its attention", li)());
    }
    const int m = (int)(pn[0] / c.hidden);
    if (m != 1 && m != 3) {
      return fail(fmt("layer {}: a norm of {} modalities", li, m)());
    }
    c.mm.push_back(m == 3 ? 1 : 0);
    c.n_q = (int)(g[0] / m);
    const int width = (int)(qkv[0] / m);
    c.n_kv = (width / c.head_dim - c.n_q) / 2;
    const bool moe = ws.has(p + "mlp.moe_mlp.gate");
    c.moe.push_back(moe ? 1 : 0);
    if (moe) {
      const auto gate = shape(p + "mlp.moe_mlp.gate");
      const auto wg = shape(p + "mlp.moe_mlp.W_gate");
      const auto s1 = shape(p + "mlp.shared_expert_fc1.weight");
      const auto m1 =
          shape(p + "mlp.modality_specific_shared_expert_fc1.weight");
      if (gate.size() != 2 || wg.size() != 3 || s1.size() != 2 ||
          m1.size() != 2) {
        return fail(fmt("layer {}: an incomplete MoE", li)());
      }
      const int dh = (int)gate[1];
      c.moe_heads = c.hidden / dh;
      c.moe_experts = (int)(gate[0] / c.moe_heads);
      c.expert_inter = (int)wg[2];
      c.shared_inter = (int)(s1[0] / 2);
      c.ms_inter = (int)(m1[0] / 6);
      if (dh != 256) { return fail("the MoE's heads are not 256 wide"); }
      if (c.moe_experts > 512) { return fail("more than 512 experts a head"); }
    } else {
      const auto up = shape(p + "mlp.up_gate_proj.weight");
      if (up.size() != 2) {
        return fail(fmt("layer {}: no MLP", li)());
      }
      c.mlp_inter = (int)(up[0] / m / 2);
    }
  }
  if (c.n_q % c.n_kv != 0) { return fail("query heads not a multiple"); }
  *out = c;
  return true;
}

// ---- tensor parallel --------------------------------------------------------

bool
MetalMagi2Dit::share_of(const Config& c, Share* out, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "MAGI-2 DiT: tensor parallel: " + m; }
    return false;
  };
  Share z;
  const int E = c.moe_experts;
  if (!c.tp.active()) {
    z.h1 = c.n_q;
    z.kv1 = c.n_kv;
    z.e1 = E;
    z.sh1 = c.shared_inter;
    z.ms1 = c.ms_inter;
    z.mlp1 = c.mlp_inter;
    *out = z;
    return true;
  }
  const TpShare& t = c.tp;
  if (t.n_heads != c.n_q || t.n_ffn != E) {
    return fail(fmt("the share counts {} heads and {} feed-forward rows "
                    "where this model has {} heads and {} experts a MoE "
                    "head (its feed-forward rows count experts)",
                    t.n_heads, t.n_ffn, c.n_q, E)());
  }
  if (t.head0 < 0 || t.head1 > c.n_q || t.head0 >= t.head1) {
    return fail(fmt("heads [{}, {}) of {}", t.head0, t.head1, c.n_q)());
  }
  if (t.ffn0 < 0 || t.ffn1 > E || t.ffn0 >= t.ffn1) {
    return fail(fmt("experts [{}, {}) of {}", t.ffn0, t.ffn1, E)());
  }
  const int g = c.n_q / std::max(1, c.n_kv);
  if (t.head0 % g != 0 || t.head1 % g != 0) {
    return fail(fmt("heads [{}, {}) split a group of {} query heads that "
                    "share a kv head", t.head0, t.head1, g)());
  }
  z.h0 = t.head0;
  z.h1 = t.head1;
  z.kv0 = t.head0 / g;
  z.kv1 = t.head1 / g;
  z.e0 = t.ffn0;
  z.e1 = t.ffn1;
  // The dense widths at the experts' fraction: every boundary a function
  // of the boundary alone, so the ranks' shares tile each width.
  auto at = [&](int f, int W) {
    return (int)((std::int64_t)f * W / std::max(1, E));
  };
  z.sh0 = at(t.ffn0, c.shared_inter);
  z.sh1 = at(t.ffn1, c.shared_inter);
  z.ms0 = at(t.ffn0, c.ms_inter);
  z.ms1 = at(t.ffn1, c.ms_inter);
  z.mlp0 = at(t.ffn0, c.mlp_inter);
  z.mlp1 = at(t.ffn1, c.mlp_inter);
  if (z.shared() <= 0 || z.ms() <= 0 || z.mlp() <= 0) {
    return fail(fmt("experts [{}, {}) leave a dense MLP's share empty",
                    t.ffn0, t.ffn1)());
  }
  *out = z;
  return true;
}

std::uint64_t
MetalMagi2Dit::expert_share_(std::uint64_t whole) const noexcept
{
  const std::uint64_t n =
      (std::uint64_t)_cfg.moe_heads * (std::uint64_t)_cfg.moe_experts;
  if (!_cfg.tp.active() || n == 0) { return whole; }
  return whole / n * (std::uint64_t)_cfg.moe_heads *
         (std::uint64_t)_sh.experts();
}

bool
MetalMagi2Dit::read_experts_(int idx, SharedBuffer dst[3]) const
{
  const MetalLlamaWeights& src = _ws->src();
  const std::uint64_t n =
      (std::uint64_t)_cfg.moe_heads * (std::uint64_t)_cfg.moe_experts;
  int i = 0;
  for (const char* w : {"W_gate", "W_up", "W_down"}) {
    const std::string nm = fmt("block.layers.{}.mlp.moe_mlp.{}", idx, w)();
    SharedBuffer& b = dst[i++];
    if (!_cfg.tp.active()) {
      if (!_ws->stream_into(nm, b.contents(), b.byte_size())) {
        return false;
      }
      continue;
    }
    // A rank's experts: each MoE head's run [e0, e1), uncached, as the
    // slots read a whole layer.
    const auto* ti = src.info(nm);
    if (ti == nullptr || n == 0 || ti->nbytes % n != 0) { return false; }
    const std::uint64_t per = ti->nbytes / n;
    const std::uint64_t run = per * (std::uint64_t)_sh.experts();
    if (b.byte_size() < run * (std::uint64_t)_cfg.moe_heads) { return false; }
    auto* d = static_cast<std::uint8_t*>(b.contents());
    for (int h = 0; h < _cfg.moe_heads; ++h) {
      const std::uint64_t at =
          ((std::uint64_t)h * _cfg.moe_experts + (std::uint64_t)_sh.e0) *
          per;
      if (!src.pread_range_into(nm, at, (std::size_t)run,
                                d + (std::uint64_t)h * run)) {
        return false;
      }
    }
  }
  return true;
}

bool
MetalMagi2Dit::tp_buffer_(SharedBuffer& b, std::size_t bytes)
{
  if (!b.empty() && b.byte_size() >= bytes) { return true; }
  // An exchange buffer stays at one address until the channel is told it
  // goes (TpChannel::forget), after what was posted from it is answered.
  if (!b.empty() && _tp_pipe != nullptr) { _tp_pipe->forget(b); }
  b = _mc->make_shared_buffer(std::max<std::size_t>(bytes, 16));
  return !b.empty();
}

std::size_t
MetalMagi2Dit::tp_exchange_bytes(const Config& c, int n, int batch)
{
  if (!c.tp.active() || n <= 0) { return 0; }
  const int lanes =
      batch >= 2 && batch <= kTpLanes ? batch : 1;
  const std::size_t pb =
      ((std::size_t)n * (std::size_t)c.hidden * 2 + 16383) / 16384 * 16384;
  return (std::size_t)lanes * 2 * pb;
}

bool
MetalMagi2Dit::begin_tp_generation(std::shared_ptr<TpChannel> ch,
                                   std::function<bool()> stop,
                                   std::string* err)
{
  auto fail = [&](const std::string& why) {
    if (err != nullptr) { *err = "tensor parallel: " + why; }
    return false;
  };
  end_tp_generation();
  if (!_cfg.tp.active()) { return fail("this model is not a rank"); }
  if (!ch) { return fail("no channel to the other ranks"); }
  auto pipe = std::make_unique<TpOverlap>(_mc, ch, kTpLanes, TpDType::kBF16);
  if (!pipe->valid()) {
    return fail("no events or thread for the exchanges");
  }
  _tp_ch = std::move(ch);
  _tp_pipe = std::move(pipe);
  _tp_seq = 0;
  // The run's stop, which the channel polls while it waits on the others.
  try {
    _tp_ch->set_stop(std::move(stop));
  } catch (...) {
  }
  return true;
}

void
MetalMagi2Dit::end_tp_generation()
{
  if (_tp_pipe == nullptr && !_tp_ch) { return; }
  // An exchange still waiting on a peer would hold the worker the join
  // below waits for: the channel is told to stop waiting first.
  if (_tp_pipe != nullptr && _tp_ch &&
      (_tp_pipe->pending() || _tp_pipe->failed())) {
    try {
      _tp_ch->set_stop([]() { return true; });
    } catch (...) {
    }
  }
  _tp_pipe.reset();
  if (_tp_ch) {
    // Every buffer the channel was handed, forgotten before it goes (a
    // transport may keep a device's registration of each).
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

// ---- load ------------------------------------------------------------------

std::unique_ptr<MetalMagi2Dit>
MetalMagi2Dit::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                    const Config& cfg, std::string* err)
{
  std::unique_ptr<MetalMagi2Dit> m(new MetalMagi2Dit());
  m->_ws = std::move(ws);
  m->_mc = mc;
  m->_cfg = cfg;
  if (!m->load_(err)) { return nullptr; }
  return m;
}

bool
MetalMagi2Dit::load_(std::string* err)
{
  auto fail = [&](const std::string& msg) {
    if (err != nullptr) { *err = "MAGI-2 DiT: " + msg; }
    return false;
  };
  _k = std::make_unique<Kernels>();
  Kernels& k = *_k;
  k.lib = _mc->load_library("magi2_bf16");
  k.elt = _mc->load_library("llm_elementwise_bf16");
  k.attn = _mc->load_library("attn_steel");
  struct Need {
    ComputeFunction* fn;
    const ComputeLibrary* lib;
    const char* name;
  };
  const Need need[] = {
      {&k.g32a16, &k.lib, "magi2_gemm32_a16"},
      {&k.g32a32, &k.lib, "magi2_gemm32_a32"},
      {&k.embed_finish, &k.lib, "magi2_embed_finish"},
      {&k.mhc_pre, &k.lib, "magi2_mhc_pre"},
      {&k.mhc_post, &k.lib, "magi2_mhc_post"},
      {&k.rms1, &k.lib, "magi2_rms1"},
      {&k.inv_rms, &k.lib, "magi2_inv_rms"},
      {&k.qk_rope, &k.lib, "magi2_qk_rope"},
      {&k.gate, &k.lib, "magi2_gate"},
      {&k.swiglu7, &k.lib, "magi2_swiglu7"},
      {&k.topk, &k.lib, "magi2_moe_topk"},
      {&k.invert, &k.lib, "magi2_moe_invert"},
      {&k.expert, &k.lib, "magi2_moe_expert"},
      {&k.combine, &k.lib, "magi2_moe_combine"},
      {&k.drop_bin, &k.lib, "magi2_moe_drop_bin"},
      {&k.add, &k.lib, "magi2_add"},
      {&k.ifill, &k.elt, "moe_ifill_i32"},
      {&k.hist, &k.elt, "moe_hist_i32"},
      {&k.sort_setup, &k.elt, "moe_sort_setup_i32"},
      {&k.scatter, &k.elt, "moe_scatter_i32"},
  };
  for (const Need& n : need) {
    *n.fn = n.lib->function(n.name);
    if (!n.fn->valid()) {
      return fail(fmt("kernel '{}' did not load", n.name)());
    }
  }
  auto sink_fc = [](int a) {
    FunctionConstants fc;
    fc.set_bool(200, (a & 1) != 0).set_bool(201, (a & 2) != 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, true)
        .set_bool(303, false);
    return fc;
  };
  for (int a = 0; a < 4; ++a) {
    k.attn_fn[a] = k.attn.function("attn_steel_h_bd128_bf16", sink_fc(a));
    if (!k.attn_fn[a].valid()) {
      return fail("the steel attention (with sinks) did not load");
    }
  }
  if (_cfg.head_dim != 128) { return fail("head_dim is not 128"); }
  // THE M5 ROUTES: matmul2d GEMMs, the matrix-core attention -- each only
  // where every entry it needs loads.
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
  if (_mc->supports_matrix_cores() && !env_on("VPIPE_MAGI2_NO_NAX_ATTN")) {
    k.nax = _mc->load_library("attn_steel_nax");
    bool all = k.nax.valid();
    for (int a = 0; a < 4 && all; ++a) {
      k.nax_fn[a] = k.nax.function("attn_steel_nax_h_bd128_bf16",
                                   sink_fc(a));
      all = k.nax_fn[a].valid();
    }
    _nax = all;
  }
  // THE ROUTED EXPERTS on the matrix cores: 64-row work items through the
  // fused gate / up, swiglu7 and down (magi2_nax.metal), where the
  // expert's width is whole chunks of 64. VPIPE_MAGI2_NO_NAX_MOE=1 keeps
  // the simdgroup kernel (the A/B).
  if (_mc->supports_matrix_cores() && !env_on("VPIPE_MAGI2_NO_NAX_MOE") &&
      _cfg.expert_inter % 64 == 0) {
    k.moe_nax = _mc->load_library("magi2_nax_bf16");
    k.expert_nax = k.moe_nax.function("magi2_moe_expert_nax");
    k.tile_items = k.elt.function("moe_tile_items_i32");
    _nax_moe = k.expert_nax.valid() && k.tile_items.valid();
  }
  // ...and elsewhere the STAGED simdgroup kernel (weights through
  // threadgroup memory), where the width is whole chunks of 64.
  // VPIPE_MAGI2_MOE_SG=1 keeps the first simdgroup kernel (the A/B).
  if (!_nax_moe && !env_on("VPIPE_MAGI2_MOE_SG") &&
      _cfg.expert_inter % 64 == 0) {
    k.expert_tg = k.lib.function("magi2_moe_expert_tg");
    _tg_moe = k.expert_tg.valid();
  }
  // THE INT8 TIER, before the projections load: a holding tier converts
  // each one as it loads.
  if (_cfg.i8_tier >= 0) { _g->set_i8(_cfg.i8_tier, _cfg.autotune); }
  // SAGEATTENTION, on the matrix-core attention alone.
  _sage_wanted =
      magi2::sage_runs(_cfg.sage, _mc->supports_matrix_cores()) && _nax;
  if (_sage_wanted && !load_sage_(err)) { return false; }
  if (_sage) {
    for (int a = 0; a < 4; ++a) {
      for (int pv = 0; pv < 2; ++pv) {
        FunctionConstants fc = sink_fc(a);
        // The sinks shifted as the smoothed keys are (the int8 QK subtracts
        // K's mean over tokens; a sink's logit must move with the scores).
        // load_sage_ took Sage only where this host's kernel does that.
        fc.set_bool(sage::kQkInt8Constant, true);
        MetalSageAttention::sink_constants(fc, _cfg.sage);
        MetalSageAttention::pv8_constants(fc, pv == 1);
        k.nax_sage[a][pv] =
            k.nax.function("attn_steel_nax_h_bd128_bf16", fc);
        if (pv == 1 && !_sage->pv8_for(_cfg.sage)) { continue; }
        if (!k.nax_sage[a][pv].valid()) {
          return fail("the int8 attention did not load");
        }
      }
    }
  }

  // A RANK's widths (tensor parallel), checked against the model's.
  {
    std::string se;
    if (!share_of(_cfg, &_sh, &se)) {
      if (err != nullptr) { *err = se; }
      return false;
    }
  }
  const bool rank = _cfg.tp.active();
  const Share& sh = _sh;
  // The wired pool the held experts are admitted through (closed with no
  // session's pool: every layer the count asks for is held, as before).
  _wire = std::make_unique<WiredPool>();
  _wire->open(_mc);
  _admit = true;
  WeightSet& ws = *_ws;
  const MetalLlamaWeights& src = ws.src();
  // COPIED, cached in the set: a shard here holds trunk and expert
  // tensors side by side, so a mapped trunk tensor would wrap its whole
  // shard -- the experts in it included -- for the GPU.
  auto t = [&](const std::string& n) {
    return ws.tensor(n, _mc, WeightSet::Residency::Copied);
  };
  auto f32_host = [&](const std::string& n) -> std::vector<float> {
    SharedBuffer b = ws.read(n, _mc, WeightSet::Residency::Copied);
    const auto* ti = src.info(n);
    std::vector<float> v;
    if (b.empty() || ti == nullptr) { return v; }
    if (ti->dtype == "F32") {
      v.resize(b.byte_size() / 4);
      std::memcpy(v.data(), b.contents(), v.size() * 4);
    } else if (ti->dtype == "BF16") {
      v.resize(b.byte_size() / 2);
      const auto* p = (const std::uint16_t*)b.contents();
      for (std::size_t i = 0; i < v.size(); ++i) {
        const std::uint32_t u = (std::uint32_t)p[i] << 16;
        std::memcpy(&v[i], &u, 4);
      }
    }
    return v;
  };
  auto upload = [&](const std::vector<float>& v) {
    SharedBuffer b =
        _mc->make_shared_buffer(std::max<std::size_t>(4, v.size() * 4));
    if (!v.empty()) { std::memcpy(b.contents(), v.data(), v.size() * 4); }
    return b;
  };
  auto need_dtype = [&](const std::string& n, const char* dt) {
    const auto* ti = src.info(n);
    return ti != nullptr && ti->dtype == dt;
  };

  _bands = f32_host("pre_adapter.rope.bands");
  const char* names[3] = {"video", "audio", "text"};
  for (int i = 0; i < 3; ++i) {
    const std::string w = fmt("pre_adapter.{}_embedder.weight", names[i])();
    const std::string b = fmt("pre_adapter.{}_embedder.bias", names[i])();
    if (!need_dtype(w, "F32") || !need_dtype(b, "F32")) {
      return fail(fmt("the {} embedder is not f32", names[i])());
    }
    _emb_w[i] = t(w);
    _emb_b[i] = t(b);
  }
  // The heads: RMSNorm (w + 1) folded into the f32 linear's columns, so
  // the GEMM over the raw row times the row's inverse RMS is the release's
  // normalize-then-project.
  const int W4 = _cfg.hidden * _cfg.streams;
  for (int i = 0; i < 2; ++i) {
    const std::vector<float> nw =
        f32_host(fmt("post_adapter.final_norm_{}.weight", names[i])());
    std::vector<float> lw =
        f32_host(fmt("post_adapter.final_linear_{}.weight", names[i])());
    if ((int)nw.size() != W4 || lw.size() % (std::size_t)W4 != 0) {
      return fail("the post-adapter's shapes");
    }
    const std::size_t rows = lw.size() / (std::size_t)W4;
    for (std::size_t r = 0; r < rows; ++r) {
      for (int c = 0; c < W4; ++c) {
        lw[r * W4 + (std::size_t)c] *= nw[(std::size_t)c] + 1.0f;
      }
    }
    _out_w[i] = upload(lw);
  }

  const int n = _cfg.streams;
  for (std::size_t li = 0; li < _cfg.layers.size(); ++li) {
    Layer L;
    L.idx = _cfg.layers[li];
    L.m = _cfg.mm[li] ? 3 : 1;
    L.moe = _cfg.moe[li] != 0;
    const std::string p = fmt("block.layers.{}.", L.idx)();
    L.attn_norm = t(p + "attention.pre_norm.weight");
    L.q_norm = t(p + "attention.q_norm.weight");
    L.k_norm = t(p + "attention.k_norm.weight");
    const int Hd = _cfg.hidden;
    const int D = _cfg.head_dim;
    const int qs = _cfg.n_q * D;
    const int kvs = _cfg.n_kv * D;
    const int ldqkv = qs + 2 * kvs;
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
      if (L.g.empty()) {
        SharedBuffer f = ws.read(p + "attention.linear_g.weight", _mc,
                                 WeightSet::Residency::Copied);
        L.g = tp_cut(_mc, f, L.m * _cfg.n_q, gr, 0, 0);
      }
      if (L.g.empty()) {
        return fail(fmt("layer {}: the gate's share", L.idx)());
      }
    }
    auto stacked = [&](const char* name, int m, int N, int K,
                       std::vector<magi2::Proj>* out) {
      std::string pe;
      if (!_g->load_stacked(ws, p + name, m, N, K, "magi2-preview/", out,
                            &pe)) {
        if (err != nullptr) { *err = "MAGI-2 DiT: " + pe; }
        return false;
      }
      return true;
    };
    // A RANK's share of a projection: its rows of each slab (an output
    // cut) or its columns (an input cut); the whole model's, the tensor.
    auto rows_of = [&](const char* name, int m, int N, int K,
                       const std::vector<std::pair<int, int>>& rows,
                       std::vector<magi2::Proj>* out) {
      if (!rank) { return stacked(name, m, N, K, out); }
      std::string pe;
      if (!_g->load_cut(ws, p + name, m, N, K, rows, 0, 0, out, &pe)) {
        if (err != nullptr) { *err = "MAGI-2 DiT: " + pe; }
        return false;
      }
      return true;
    };
    auto cols_of = [&](const char* name, int m, int N, int K, int c0,
                       int c1, std::vector<magi2::Proj>* out) {
      if (!rank) { return stacked(name, m, N, K, out); }
      std::string pe;
      if (!_g->load_cut(ws, p + name, m, N, K, {{0, N}}, c0, c1, out,
                        &pe)) {
        if (err != nullptr) { *err = "MAGI-2 DiT: " + pe; }
        return false;
      }
      return true;
    };
    if (!rows_of("attention.linear_qkv.weight", L.m, ldqkv, Hd,
                 {{sh.h0 * D, sh.h1 * D},
                  {qs + sh.kv0 * D, qs + sh.kv1 * D},
                  {qs + kvs + sh.kv0 * D, qs + kvs + sh.kv1 * D}},
                 &L.qkv) ||
        !cols_of("attention.linear_proj.weight", L.m, Hd, qs, sh.h0 * D,
                 sh.h1 * D, &L.proj)) {
      return false;
    }
    {
      std::vector<float> s = f32_host(p + "attention.sinks");
      if (rank && (int)s.size() == _cfg.n_q) {
        s = std::vector<float>(s.begin() + sh.h0, s.begin() + sh.h1);
      }
      std::vector<std::uint16_t> h(s.size());
      for (std::size_t i = 0; i < s.size(); ++i) { h[i] = to_bf16(s[i]); }
      L.sinks = _mc->make_shared_buffer(std::max<std::size_t>(4, h.size() * 2));
      std::memcpy(L.sinks.contents(), h.data(), h.size() * 2);
    }
    L.mlp_norm = t(p + "mlp.pre_norm.weight");
    if (L.moe) {
      if (!stacked("mlp.split_linear.weight", 1, Hd, Hd, &L.split) ||
          !stacked("mlp.merge_linear.weight", 1, Hd, Hd, &L.merge)) {
        return false;
      }
      L.rgate = t(p + "mlp.moe_mlp.gate");
      // The release routes with the EMA bias (its loader copies it over
      // expert_bias): the weights are the EMA model's.
      L.rbias = t(p + "mlp.moe_mlp.router.expert_bias_ema");
      std::uint64_t eb = 0;
      for (const char* w : {"W_gate", "W_up", "W_down"}) {
        const auto* ti = src.info(p + "mlp.moe_mlp." + w);
        if (ti == nullptr) { return fail("an expert tensor is missing"); }
        eb += expert_share_(ti->nbytes);
      }
      if (_expert_bytes != 0 && eb != _expert_bytes) {
        return fail("MoE layers of different expert sizes");
      }
      _expert_bytes = eb;
      // HELD while the count asks AND the wired pool admits a layer:
      // past the pool a held layer would sit unprotected for the
      // compressor to take (WiredPool::wirable), so it streams instead,
      // and every MoE layer after the first refusal with it.
      L.streamed = (_cfg.resident_moe >= 0 &&
                    _moe_held >= _cfg.resident_moe) ||
                   !_admit || (_wire->on() && !_wire->wirable(eb));
      if (L.streamed) {
        _admit = false;
        ++_moe_streamed;
      } else {
        // Held: read once, UNCACHED (pread, F_NOCACHE -- the slots'
        // route), into buffers of the model's own. Copied through the
        // shard's mapping it would pass every byte through the page
        // cache -- the whole expert stack on a box that holds it -- and a
        // mapped tensor would fault its pages per use.
        // A rank reads its run of each head's experts alone.
        int i = 0;
        for (const char* w : {"W_gate", "W_up", "W_down"}) {
          const auto* ti = src.info(p + "mlp.moe_mlp." + w);
          L.wx[i++] = _mc->make_shared_buffer(expert_share_(ti->nbytes));
        }
        const bool got = !L.wx[0].empty() && !L.wx[1].empty() &&
                         !L.wx[2].empty() && read_experts_(L.idx, L.wx);
        if (!got && rank) {
          return fail(fmt("layer {}: the share's experts did not read",
                          L.idx)());
        }
        if (!got) {
          // The whole tensors, through the set's read.
          i = 0;
          for (const char* w : {"W_gate", "W_up", "W_down"}) {
            L.wx[i++] = ws.read(p + "mlp.moe_mlp." + w, _mc,
                                WeightSet::Residency::Copied);
          }
        }
        if (L.wx[0].empty() || L.wx[1].empty() || L.wx[2].empty()) {
          return fail(fmt("layer {}: the experts did not read", L.idx)());
        }
        if (_wire->on()) {
          SharedBuffer* bufs[3] = {&L.wx[0], &L.wx[1], &L.wx[2]};
          const std::size_t got = _wire->wire_set(bufs, true);
          _wire->note_wired(_mc, got, eb);
          // A refusal keeps this layer (read, partly protected) and
          // streams the rest.
          if (got < eb && !_wire->wirable(eb)) { _admit = false; }
        }
        ++_moe_held;
      }
      // The shared and modality experts' fused gate | up rows are
      // interleaved (swiglu7 reads x[2 j], x[2 j + 1]): a share of the
      // width is one run of rows.
      const int sw = _cfg.shared_inter, ms = _cfg.ms_inter;
      if (!rows_of("mlp.shared_expert_fc1.weight", 1, 2 * sw, Hd,
                   {{2 * sh.sh0, 2 * sh.sh1}}, &L.sh1) ||
          !cols_of("mlp.shared_expert_fc2.weight", 1, Hd, sw, sh.sh0,
                   sh.sh1, &L.sh2) ||
          !rows_of("mlp.modality_specific_shared_expert_fc1.weight", 3,
                   2 * ms, Hd, {{2 * sh.ms0, 2 * sh.ms1}}, &L.ms1) ||
          !cols_of("mlp.modality_specific_shared_expert_fc2.weight", 3, Hd,
                   ms, sh.ms0, sh.ms1, &L.ms2)) {
        return false;
      }
      if (!need_dtype(p + "mlp.moe_mlp.gate", "F32") ||
          !need_dtype(p + "mlp.moe_mlp.W_gate", "BF16") ||
          !need_dtype(p + "mlp.moe_mlp.W_up", "BF16") ||
          !need_dtype(p + "mlp.moe_mlp.W_down", "BF16")) {
        return fail(fmt("layer {}: the MoE's dtypes", L.idx)());
      }
    } else {
      const int I = _cfg.mlp_inter;
      if (!rows_of("mlp.up_gate_proj.weight", L.m, 2 * I, Hd,
                   {{2 * sh.mlp0, 2 * sh.mlp1}}, &L.up) ||
          !cols_of("mlp.down_proj.weight", L.m, Hd, I, sh.mlp0, sh.mlp1,
                   &L.down)) {
        return false;
      }
    }
    // Hyper-connections.
    const std::vector<float> nw = f32_host(p + "mhc_norm.weight");
    L.mhc_m = (int)(nw.size() / (std::size_t)W4);
    if (L.mhc_m != 1 && L.mhc_m != 3) {
      return fail(fmt("layer {}: the mHC norm's shape", L.idx)());
    }
    const int nh = 2 * n + n * n;
    for (int part = 0; part < 2; ++part) {
      const char* nm = part == 0 ? "attn" : "mlp";
      const std::vector<float> phi = f32_host(p + "mhc_phi_fused_" + nm);
      if ((int)phi.size() != W4 * nh) {
        return fail(fmt("layer {}: mhc_phi_fused_{}'s shape", L.idx, nm)());
      }
      std::vector<float> fold((std::size_t)L.mhc_m * nh * W4);
      for (int mm = 0; mm < L.mhc_m; ++mm) {
        for (int j = 0; j < nh; ++j) {
          for (int c = 0; c < W4; ++c) {
            fold[((std::size_t)mm * nh + j) * W4 + c] =
                phi[(std::size_t)c * nh + j] *
                (nw[(std::size_t)mm * W4 + c] + 1.0f);
          }
        }
      }
      std::vector<float> ab;
      ab.push_back(f32_host(p + "mhc_alpha_pre_" + nm).at(0));
      ab.push_back(f32_host(p + "mhc_alpha_post_" + nm).at(0));
      ab.push_back(f32_host(p + "mhc_alpha_res_" + nm).at(0));
      for (const char* bn : {"pre", "post", "res"}) {
        const std::vector<float> b =
            f32_host(p + "mhc_bias_" + bn + "_" + nm);
        ab.insert(ab.end(), b.begin(), b.end());
      }
      if ((int)ab.size() != 3 + 2 * n + n * n) {
        return fail(fmt("layer {}: the mHC biases' shapes", L.idx)());
      }
      (part == 0 ? L.phi_attn : L.phi_mlp) = upload(fold);
      (part == 0 ? L.ab_attn : L.ab_mlp) = upload(ab);
    }
    _layers.push_back(std::move(L));
  }
  if (_moe_streamed > 0) {
    // Two slots, each a layer's three expert tensors (every MoE layer's
    // are the same shapes: checked above).
    int first = -1;
    for (const Layer& ly : _layers) {
      if (ly.moe) { first = ly.idx; break; }
    }
    for (auto& sl : _slot) {
      int i = 0;
      for (const char* w : {"W_gate", "W_up", "W_down"}) {
        const auto* ti =
            src.info(fmt("block.layers.{}.mlp.moe_mlp.{}", first, w)());
        sl[i] = _mc->make_shared_buffer(expert_share_(ti->nbytes));
        if (sl[i].empty()) { return fail("the expert slots"); }
        ++i;
      }
    }
  }
  _stream = _mc->make_command_stream();
  return true;
}

// ---- scratch ---------------------------------------------------------------

namespace {

// THE SCRATCH, one list: what ensure_scratch_ allocates for n tokens, in
// the order it does, so the plan's figure is the allocation's.
struct ScratchSizes {
  std::size_t S, D, coef, Y, A, qkv, g, O, P, U, act, sin, cos, emb;
  std::size_t split, logits, eid, prob, hist, boff, cursor, tile2e, nt,
      srow, sdst, pos, yexp, comb, merge, x1, x2, a1, a2, y1, y2, rs, vel,
      items;
  int chunk;
  std::size_t total() const
  {
    return S + D + coef + Y + A + qkv + g + O + P + U + act + sin + cos +
           emb + split + logits + eid + prob + hist + boff + cursor +
           tile2e + nt + srow + sdst + pos + yexp + comb + merge + x1 + x2 +
           a1 + a2 + y1 + y2 + rs + vel + items;
  }
};

// A config's widths as its model runs them: a rank's share, the whole
// model's when it is not split (or the share does not fit: never loaded).
MetalMagi2Dit::Share
widths_(const MetalMagi2Dit::Config& c)
{
  MetalMagi2Dit::Share w;
  if (!MetalMagi2Dit::share_of(c, &w, nullptr)) {
    MetalMagi2Dit::Config whole = c;
    whole.tp = TpShare{};
    MetalMagi2Dit::share_of(whole, &w, nullptr);
  }
  return w;
}

ScratchSizes
scratch_sizes_(const MetalMagi2Dit::Config& c, int n, int maxm)
{
  ScratchSizes z{};
  const MetalMagi2Dit::Share w = widths_(c);
  // Whole 16 KB pages, as the device holds a buffer.
  auto b = [](std::size_t bytes) {
    return (std::max<std::size_t>(bytes, 16) + 16383) / 16384 * 16384;
  };
  const std::size_t H = (std::size_t)c.hidden, W4 = H * c.streams;
  const std::size_t qs = (std::size_t)w.heads() * c.head_dim;
  const std::size_t kvs = (std::size_t)w.kv() * c.head_dim;
  const int nh = 2 * c.streams + c.streams * c.streams;
  const std::size_t N = (std::size_t)n;
  z.S = b(N * W4 * 2);
  z.D = b(N * nh * 4);
  z.coef = b(N * (c.streams + c.streams * c.streams) * 4);
  z.Y = b(N * H * 2);
  z.A = b(N * H * 2);
  z.qkv = b(N * (qs + 2 * kvs) * 2);
  z.g = b(N * w.heads() * 2);
  z.O = b(N * qs * 2);
  z.P = b(N * H * 2);
  const std::size_t I = std::max((std::size_t)w.mlp(), H);
  z.U = b(N * 2 * I * 2);
  z.act = b(N * I * 2);
  const int R = 3 * c.head_dim / 8;
  z.sin = b(N * R * 4);
  z.cos = b(N * R * 4);
  // The embedders run a span at a time in chunks of rows.
  z.emb = b((std::size_t)4096 * W4 * 4);
  // The MoE runs a chunk of tokens at a time: its expert outputs are f32
  // [pairs, 256], kept under ~512 MB (VPIPE_MAGI2_MOE_CHUNK_MB the A/B).
  const int per_tok = c.moe_heads * c.top_k;
  std::size_t budget = (std::size_t)512 << 20;
  if (const char* e = std::getenv("VPIPE_MAGI2_MOE_CHUNK_MB")) {
    if (std::atoi(e) > 0) { budget = (std::size_t)std::atoi(e) << 20; }
  }
  z.chunk = std::max(1, (int)std::min<std::size_t>(
                            (std::size_t)n,
                            budget / (256 * 4) / (std::size_t)per_tok));
  const std::size_t np = (std::size_t)z.chunk * per_tok;
  const int E = c.moe_heads * c.moe_experts;
  const std::size_t maxtiles = (np + maxm - 1) / maxm + (std::size_t)E;
  const std::size_t npad = maxtiles * maxm;
  z.split = b(N * H * 2);
  z.logits = b((std::size_t)z.chunk * E * 4);
  z.eid = b(np * 4);
  z.prob = b(np * 4);
  z.hist = b((std::size_t)E * 4);
  z.boff = b((std::size_t)E * 4);
  z.cursor = b((std::size_t)E * 4);
  z.tile2e = b(maxtiles * 4);
  z.items = b(maxtiles * 4);   // the matrix-core kernel's work items
  z.nt = b(16);
  z.srow = b(npad * 4);
  z.sdst = b(npad * 4);
  z.pos = b(np * 4);
  z.yexp = b(npad * 256 * 4);
  z.comb = b(N * H * 2);
  z.merge = b(N * H * 2);
  z.x1 = b(N * 2 * w.shared() * 2);
  z.x2 = b(N * 2 * w.ms() * 2);
  z.a1 = b(N * w.shared() * 2);
  z.a2 = b(N * w.ms() * 2);
  z.y1 = b(N * H * 2);
  z.y2 = b(N * H * 2);
  z.rs = b(N * 4);
  z.vel = b(N * std::max(c.video_c, c.audio_c) * 4);
  return z;
}

// The block projections the int8 tier runs, over n tokens.
std::vector<magi2::Gemms::Shape>
i8_shapes_(const MetalMagi2Dit::Config& c)
{
  const MetalMagi2Dit::Share w = widths_(c);
  const int H = c.hidden;
  const int qs = w.heads() * c.head_dim;
  const int ldqkv = qs + 2 * w.kv() * c.head_dim;
  return {{ldqkv, H, 0},       {H, qs, 0},
          {2 * w.mlp(), H, 0}, {H, w.mlp(), 0},
          {H, H, 0},           {2 * w.shared(), H, 0},
          {H, w.shared(), 0},  {2 * w.ms(), H, 0},
          {H, w.ms(), 0}};
}

}  // namespace

std::size_t
MetalMagi2Dit::scratch_bytes(const Config& c, int n, int batch, int inputs,
                             bool matrix_cores)
{
  if (n <= 0) { return 0; }
  const ScratchSizes z = scratch_sizes_(c, n, kMaxm);
  const std::size_t W4 = (std::size_t)c.hidden * c.streams;
  const int R = 3 * c.head_dim / 8;
  // Each sample past the first: its residual stream and rope tables; and
  // every sample's input rows (`inputs` elements a sample, f32).
  auto paged = [](std::size_t b) { return (b + 16383) / 16384 * 16384; };
  const std::size_t extra =
      (std::size_t)std::max(0, batch - 1) *
          (paged((std::size_t)n * W4 * 2) +
           2 * paged((std::size_t)n * R * 4)) +
      paged((std::size_t)std::max(1, batch) *
            ((std::size_t)inputs * 4 + 6 * 256));
  // The int8 context's scratch over the block projections (reserved at
  // the top of a forward).
  const std::size_t i8 = magi2::Gemms::i8_scratch_bytes(
      matrix_cores ? c.i8_tier : -1, c.autotune, n, i8_shapes_(c));
  // SageAttention's prologue: Q and K (and V) in int8 over the longest
  // sample (one scratch for the batch, its samples one after another).
  std::size_t sg = 0;
  if (magi2::sage_runs(c.sage, matrix_cores)) {
    const Share w = widths_(c);
    sg = MetalSageAttention::private_bytes(
        w.heads(), w.kv(), n, n, c.head_dim,
        MetalSageAttention::nax_query_block(),
        MetalSageAttention::nax_key_block(), 0, 0, c.sage.pv8);
  }
  // A rank's exchange buffers, and each sample's hyper-connection
  // coefficients past the first (kept to its post step).
  const std::size_t coef =
      (std::size_t)std::max(0, batch - 1) * paged(z.coef);
  return z.total() + extra + i8 + sg + coef +
         tp_exchange_bytes(c, n, batch);
}

bool
MetalMagi2Dit::ensure_scratch_(int n, std::string* err)
{
  (void)err;
  if (_s && _s->n >= n) { return true; }
  _s = std::make_unique<Scratch>();
  Scratch& s = *_s;
  s.n = n;
  const ScratchSizes z = scratch_sizes_(_cfg, n, kMaxm);
  auto mk = [&](std::size_t bytes) { return _mc->make_shared_buffer(bytes); };
  s.S = mk(z.S); s.D = mk(z.D); s.coef = mk(z.coef); s.Y = mk(z.Y);
  s.A = mk(z.A); s.qkv = mk(z.qkv); s.g = mk(z.g); s.O = mk(z.O);
  s.P = mk(z.P); s.U = mk(z.U); s.act = mk(z.act); s.sin = mk(z.sin);
  s.cos = mk(z.cos); s.emb = mk(z.emb);
  s.chunk = z.chunk;
  s.split = mk(z.split); s.logits = mk(z.logits); s.eid = mk(z.eid);
  s.prob = mk(z.prob); s.hist = mk(z.hist); s.boff = mk(z.boff);
  s.cursor = mk(z.cursor); s.tile2e = mk(z.tile2e); s.nt = mk(z.nt);
  s.srow = mk(z.srow); s.sdst = mk(z.sdst); s.pos = mk(z.pos);
  s.items = mk(z.items);
  s.yexp = mk(z.yexp); s.comb = mk(z.comb); s.merge = mk(z.merge);
  s.x1 = mk(z.x1); s.x2 = mk(z.x2); s.a1 = mk(z.a1); s.a2 = mk(z.a2);
  s.y1 = mk(z.y1); s.y2 = mk(z.y2); s.rs = mk(z.rs); s.vel = mk(z.vel);
  return true;
}

void
MetalMagi2Dit::release_scratch()
{
  _s.reset();
  if (_g) { _g->release_scratch(); }
  // SageAttention's prologue scratch with it: the object is made again at
  // the next forward.
  _sage.reset();
}

bool
MetalMagi2Dit::load_sage_(std::string* err)
{
  bool fatal = false;
  // Every attention here has sinks: on a host whose flash kernel cannot
  // take them under the int8 QK, Sage declines and attention stays dense.
  _sage = MetalSageAttention::load_for_model(_mc, /*bf16=*/true, _cfg.sage,
                                             "magi2 preview", &fatal,
                                             /*with_sinks=*/true);
  if (fatal || !_sage) {
    if (fatal && err != nullptr) {
      *err = "MAGI-2 DiT: SageAttention's kernels did not load";
    }
    _sage_wanted = false;
    return !fatal;
  }
  _sage->reserve_pv8(_sage->pv8_for(_cfg.sage));
  return true;
}

bool
MetalMagi2Dit::uses_mma() const noexcept
{
  return _g && _g->matrix_cores();
}

int
MetalMagi2Dit::i8_tier() const noexcept
{
  return _g ? _g->i8_tier() : -1;
}

long
MetalMagi2Dit::i8_gemms() const noexcept
{
  return _g ? _g->i8_gemms() : 0;
}

long
MetalMagi2Dit::dense_gemms() const noexcept
{
  return _g ? _g->dense_gemms() : 0;
}

std::size_t
MetalMagi2Dit::resident_bytes() const
{
  std::size_t t = 0;
  auto add = [&](const SharedBuffer& b) { t += b.byte_size(); };
  for (const Layer& ly : _layers) {
    for (const SharedBuffer* b :
         {&ly.attn_norm, &ly.q_norm, &ly.k_norm, &ly.g, &ly.sinks,
          &ly.mlp_norm, &ly.rgate, &ly.rbias, &ly.wx[0], &ly.wx[1],
          &ly.wx[2], &ly.phi_attn, &ly.phi_mlp, &ly.ab_attn, &ly.ab_mlp}) {
      add(*b);
    }
    for (const auto* ps : {&ly.qkv, &ly.proj, &ly.up, &ly.down, &ly.split,
                           &ly.merge, &ly.sh1, &ly.sh2, &ly.ms1, &ly.ms2}) {
      for (const magi2::Proj& pj : *ps) { t += pj.bytes(); }
    }
  }
  for (int i = 0; i < 3; ++i) {
    add(_emb_w[i]);
    add(_emb_b[i]);
  }
  add(_out_w[0]);
  add(_out_w[1]);
  for (const auto& sl : _slot) {
    for (const SharedBuffer& b : sl) { add(b); }
  }
  return t;
}

std::size_t
MetalMagi2Dit::trunk_bytes(WeightSet& ws, const Config& c)
{
  // Every tensor the load keeps but the routed experts, as stored --
  // plus what it builds wider: the per-modality hyper-connection folds
  // ([m][24, 4H] f32 against the stored [4H, 24] once) and the heads'
  // folded linears (f32, as stored).
  const MetalLlamaWeights& src = ws.src();
  std::size_t t = 0;
  for (const std::string& nm : src.tensor_names()) {
    if (nm.find(".moe_mlp.W_") != std::string::npos ||
        nm.find("expert_bias") != std::string::npos ||
        nm.find("mhc_alpha_") != std::string::npos ||
        nm.find("mhc_bias_") != std::string::npos ||
        nm.find("mhc_norm.") != std::string::npos ||
        nm.find("final_norm_") != std::string::npos ||
        nm.find("rope.bands") != std::string::npos) {
      continue;
    }
    if (const auto* ti = src.info(nm)) { t += ti->nbytes; }
  }
  // The EMA bias is the one kept (a copy of each MoE layer's).
  const std::size_t E = (std::size_t)c.moe_heads * c.moe_experts;
  int nmoe = 0, nmm = 0;
  for (std::size_t i = 0; i < c.layers.size(); ++i) {
    nmoe += c.moe[i] ? 1 : 0;
    nmm += c.mm[i] ? 1 : 0;
  }
  t += (std::size_t)nmoe * E * 4;
  const std::size_t W4 = (std::size_t)c.hidden * c.streams;
  const std::size_t nh = 2 * (std::size_t)c.streams +
                         (std::size_t)c.streams * c.streams;
  // A layer's two folds: m copies of [24, 4H] where the file holds one;
  // the alpha | bias vectors beside them.
  t += (std::size_t)nmm * 2 * 2 * nh * W4 * 4;
  t += c.layers.size() * 2 * (3 + nh) * 4;
  // The block projections as the load holds them: a rank's share of each
  // (Share), at the layout the tier holds them in where it holds one --
  // the bf16 as stored out, what is held in. And a rank's heads of the
  // gate and the sinks (the sinks as stored, as the whole model's are
  // counted).
  const Share w = widths_(c);
  if (c.i8_tier >= 1 || c.tp.active()) {
    const int Hd = c.hidden, D = c.head_dim;
    const int qs = c.n_q * D, qsl = w.heads() * D;
    const int ldqkv = qs + 2 * c.n_kv * D;
    const int ldl = qsl + 2 * w.kv() * D;
    std::size_t bf = 0, held = 0;
    auto proj = [&](int m, int N, int K, int Nl, int Kl) {
      bf += (std::size_t)m * magi2::proj_bytes(N, K, -1);
      held += (std::size_t)m * magi2::proj_bytes(Nl, Kl, c.i8_tier);
    };
    for (std::size_t i = 0; i < c.layers.size(); ++i) {
      const int m = c.mm[i] ? 3 : 1;
      proj(m, ldqkv, Hd, ldl, Hd);
      proj(m, Hd, qs, Hd, qsl);
      bf += (std::size_t)m * c.n_q * Hd * 2 + (std::size_t)c.n_q * 4;
      held += (std::size_t)m * w.heads() * Hd * 2 +
              (std::size_t)w.heads() * 4;
      if (c.moe[i]) {
        proj(1, Hd, Hd, Hd, Hd);
        proj(1, Hd, Hd, Hd, Hd);
        proj(1, 2 * c.shared_inter, Hd, 2 * w.shared(), Hd);
        proj(1, Hd, c.shared_inter, Hd, w.shared());
        proj(3, 2 * c.ms_inter, Hd, 2 * w.ms(), Hd);
        proj(3, Hd, c.ms_inter, Hd, w.ms());
      } else {
        proj(m, 2 * c.mlp_inter, Hd, 2 * w.mlp(), Hd);
        proj(m, Hd, c.mlp_inter, Hd, w.mlp());
      }
    }
    t = t - bf + held;
  }
  return t;
}

void
MetalMagi2Dit::force_routes(int idx, std::vector<int> picks)
{
  for (auto it = _forced.begin(); it != _forced.end(); ++it) {
    if (it->first == idx) { _forced.erase(it); break; }
  }
  if (picks.empty()) { return; }
  SharedBuffer b = _mc->make_shared_buffer(picks.size() * 4);
  std::memcpy(b.contents(), picks.data(), picks.size() * 4);
  _forced.emplace_back(idx, std::move(b));
}

std::vector<int>
MetalMagi2Dit::routes(int idx) const
{
  for (const auto& k : _kept) {
    if (k.first == idx) { return k.second; }
  }
  return {};
}

// ---- forward ---------------------------------------------------------------

bool
MetalMagi2Dit::forward(const Sample& smp, std::vector<float>* vel_video,
                       std::vector<float>* vel_audio, std::string* err)
{
  std::vector<std::vector<float>> vv, va;
  if (!forward(std::vector<const Sample*>{&smp}, &vv, &va, err)) {
    return false;
  }
  *vel_video = std::move(vv[0]);
  *vel_audio = std::move(va[0]);
  return true;
}

bool
MetalMagi2Dit::forward(const std::vector<const Sample*>& batch,
                       std::vector<std::vector<float>>* vel_video,
                       std::vector<std::vector<float>>* vel_audio,
                       std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "MAGI-2 DiT: " + m; }
    return false;
  };
  const Config& c = _cfg;
  const Kernels& k = *_k;
  const int nb = (int)batch.size();
  if (nb <= 0) { return fail("an empty batch"); }
  std::vector<Layout> Ls;
  int nmax = 0;
  for (const Sample* p : batch) {
    Ls.push_back(layout(*p, _bands));
    if (Ls.back().n <= 0) { return fail("an empty sequence"); }
    nmax = std::max(nmax, Ls.back().n);
  }
  if (!ensure_scratch_(nmax, err)) { return false; }
  Scratch& s = *_s;
  const int H = c.hidden, W4 = H * c.streams;
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
  const int R = 3 * c.head_dim / 8;
  const int nh = 2 * c.streams + c.streams * c.streams;
  // THE BATCH. Each sample keeps its own residual stream and rope table;
  // sample b > 0's are swapped into the scratch's S / sin / cos while it
  // runs (enter / leave), so every encoder below binds the sample's own.
  // The rest of the scratch is shared: the samples run one after another.
  for (int b = 1; b < nb; ++b) {
    if ((int)s.bS.size() < b) {
      s.bS.emplace_back();
      s.bsin.emplace_back();
      s.bcos.emplace_back();
      s.bcoef.emplace_back();
    }
    auto grow = [&](SharedBuffer& buf, std::size_t bytes) {
      if (buf.byte_size() < bytes) { buf = _mc->make_shared_buffer(bytes); }
      return !buf.empty();
    };
    const std::size_t N = (std::size_t)s.n;
    if (!grow(s.bS[(std::size_t)b - 1], N * W4 * 2) ||
        !grow(s.bsin[(std::size_t)b - 1], N * R * 4) ||
        !grow(s.bcos[(std::size_t)b - 1], N * R * 4) ||
        !grow(s.bcoef[(std::size_t)b - 1], s.coef.byte_size())) {
      return fail("a batch sample's residual stream");
    }
  }
  // ...and its hyper-connection's coefficients: a rank runs a sample's
  // sublayer, then the next sample's, before the first's post step.
  auto swap_in = [&](int b) {
    if (b == 0) { return; }
    std::swap(s.S, s.bS[(std::size_t)b - 1]);
    std::swap(s.sin, s.bsin[(std::size_t)b - 1]);
    std::swap(s.cos, s.bcos[(std::size_t)b - 1]);
    std::swap(s.coef, s.bcoef[(std::size_t)b - 1]);
  };
  // The sample in the scratch: its token count and modality spans --
  // rows [0, nv) video, [nv, nv + na) audio, the rest text.
  int n = 0;
  int span0[3] = {0, 0, 0}, spann[3] = {0, 0, 0};
  auto enter = [&](int b) {
    swap_in(b);
    const Layout& L = Ls[(std::size_t)b];
    n = L.n;
    span0[0] = 0; span0[1] = L.nv; span0[2] = L.nv + L.na;
    spann[0] = L.nv; spann[1] = L.na; spann[2] = L.nt;
  };
  auto leave = [&](int b) { swap_in(b); };
  for (int b = 0; b < nb; ++b) {
    enter(b);
    const Layout& L = Ls[(std::size_t)b];
    std::memcpy(s.sin.contents(), L.sin.data(), L.sin.size() * 4);
    std::memcpy(s.cos.contents(), L.cos.data(), L.cos.size() * 4);
    leave(b);
  }

  // With no encoder open: the int8 tier's split-K tuning of the shapes the
  // last forward met, and SageAttention's scratch at the batch's longest
  // sample (built once while it stays put). A batch shares one step.
  magi2::Gemms& gm = *_g;
  gm.tune_pending();
  if (!gm.reserve(nmax, i8_shapes_(c))) { return fail("the int8 scratch"); }
  if (_sage_wanted && !_sage && !load_sage_(err)) { return false; }
  const bool pv8_step = _sage != nullptr && _sage->pv8_for(c.sage) &&
                        magi2::sage_pv8_at(c.sage, batch[0]->step);
  if (_sage) {
    std::string se;
    if (!_sage->reserve(nq, nkv, nmax, nmax, c.head_dim,
                        MetalSageAttention::nax_query_block(),
                        MetalSageAttention::nax_key_block(),
                        _sage->pv8_for(c.sage), &se)) {
      return fail("SageAttention: " + se);
    }
  }

  // THE METER: what an aborted forward counted and never flushed goes.
  _flops.begin();
  ComputeEncoder enc = _stream.begin_compute();
  // THE STREAMED EXPERTS. Reads into the two slots, one outstanding at a
  // time per slot; slot j of streamed layer q is q % 2. A read is issued
  // once the slot's previous user has finished on the GPU (its fence was
  // waited), so the first two go at once and every later one right after
  // the commit of the layer before it -- under that layer's GPU work.
  std::vector<int> streamed;              // indices into _layers
  for (std::size_t i = 0; i < _layers.size(); ++i) {
    if (_layers[i].streamed) { streamed.push_back((int)i); }
  }
  std::future<bool> reads[2];
  auto read_into = [&](int q) {
    const Layer& ly = _layers[(std::size_t)streamed[(std::size_t)q]];
    const int slot = q % 2;
    reads[slot] = std::async(std::launch::async, [this, &ly, slot]() {
      return read_experts_(ly.idx, _slot[slot]);
    });
  };
  auto join_reads = [&]() {
    for (auto& r : reads) {
      if (r.valid()) { r.wait(); }
    }
  };
  for (int q = 0; q < (int)streamed.size() && q < 2; ++q) { read_into(q); }
  int next_read = std::min<int>(2, (int)streamed.size());
  int streamed_seen = 0;
  auto g32 = [&](bool a16, const SharedBuffer& A, std::size_t a_off, int lda,
                 const SharedBuffer& B, std::size_t b_off, int ldb,
                 const SharedBuffer* bias, const SharedBuffer* rs,
                 std::size_t rs_off, const SharedBuffer& C, std::size_t c_off,
                 int ldc, int M, int N, int K, int Z = 1, long az = 0,
                 long bz = 0, long cz = 0) {
    if (M <= 0 || N <= 0) { return; }
    _flops.add((std::uint64_t)Z * gemm_flops(M, N, K));
    G32 p{M, N, K, lda, ldb, ldc, az, bz, cz, bias != nullptr ? 1 : 0,
          rs != nullptr ? 1 : 0};
    enc.set_function(a16 ? k.g32a16 : k.g32a32);
    enc.set_buffer(0, A, a_off);
    enc.set_buffer(1, B, b_off);
    enc.set_buffer(2, bias != nullptr ? *bias : B);
    enc.set_buffer(3, rs != nullptr ? *rs : B, rs_off);
    enc.set_buffer(4, C, c_off);
    enc.set_constant_bytes(5, &p, sizeof p);
    enc.dispatch({(unsigned)((N + 31) / 32 * 128), (unsigned)((M + 31) / 32),
                  (unsigned)Z},
                 {128, 1, 1});
  };
  // A per-modality projection: one per modality, each over its span's
  // rows (m == 1: one over all of them), through its route -- int8 where
  // the tier takes it.
  auto mm_proj = [&](int m, const SharedBuffer& x,
                     const std::vector<magi2::Proj>& ps,
                     const SharedBuffer& y) {
    if (m == 1) {
      gm.proj(enc, x, 0, ps[0], y, 0, n);
      return;
    }
    for (int i = 0; i < 3; ++i) {
      const magi2::Proj& pj = ps[(std::size_t)i];
      gm.proj(enc, x, (std::size_t)span0[i] * pj.k * 2, pj, y,
              (std::size_t)span0[i] * pj.n * 2, spann[i]);
    }
  };
  auto rms_rows = [&](int m, const SharedBuffer& x, const SharedBuffer& y,
                      const SharedBuffer& w, int D) {
    for (int i = 0; i < 3; ++i) {
      const int r0 = m == 1 ? (i == 0 ? 0 : n) : span0[i];
      const int rn = m == 1 ? (i == 0 ? n : 0) : spann[i];
      if (rn <= 0) { continue; }
      enc.set_function(k.rms1);
      enc.set_buffer(0, x, (std::size_t)r0 * D * 2);
      enc.set_buffer(1, y, (std::size_t)r0 * D * 2);
      enc.set_buffer(2, w, (std::size_t)(m == 1 ? 0 : i) * D * 4);
      enc.set_constant(3, D);
      enc.set_constant(4, c.eps);
      enc.dispatch({(unsigned)(256 * rn), 1, 1}, {256, 1, 1});
    }
  };
  auto swiglu = [&](const SharedBuffer& x, const SharedBuffer& y, int I,
                    int rows) {
    enc.set_function(k.swiglu7);
    enc.set_buffer(0, x);
    enc.set_buffer(1, y);
    enc.set_constant(2, I);
    enc.dispatch({(unsigned)I, (unsigned)rows, 1}, {64, 1, 1});
  };
  auto add = [&](const SharedBuffer& a, const SharedBuffer& b,
                 const SharedBuffer& y, std::size_t count) {
    enc.set_function(k.add);
    enc.set_buffer(0, a);
    enc.set_buffer(1, b);
    enc.set_buffer(2, y);
    enc.set_constant(3, (unsigned)count);
    enc.dispatch({(unsigned)count, 1, 1}, {256, 1, 1});
  };
  // The hyper-connection's pre step: phi dots per modality, then the
  // coefficients and the sublayer's input.
  auto mhc_pre = [&](const Layer& ly, const SharedBuffer& phi,
                     const SharedBuffer& ab) {
    for (int i = 0; i < 3; ++i) {
      const int r0 = ly.mhc_m == 1 ? (i == 0 ? 0 : n) : span0[i];
      const int rn = ly.mhc_m == 1 ? (i == 0 ? n : 0) : spann[i];
      g32(true, s.S, (std::size_t)r0 * W4 * 2, W4, phi,
          (std::size_t)(ly.mhc_m == 1 ? 0 : i) * nh * W4 * 4, W4, nullptr,
          nullptr, 0, s.D, (std::size_t)r0 * nh * 4, nh, rn, nh, W4);
    }
    enc.set_function(k.mhc_pre);
    enc.set_buffer(0, s.S);
    enc.set_buffer(1, s.D);
    enc.set_buffer(2, s.coef);
    enc.set_buffer(3, s.Y);
    enc.set_buffer(4, ab);
    enc.set_constant(5, H);
    enc.set_constant(6, c.streams);
    enc.set_constant(7, nh);
    enc.set_constant(8, c.eps);
    enc.set_constant(9, c.sk_iters);
    enc.set_constant(10, c.sk_eps);
    enc.dispatch({(unsigned)(256 * n), 1, 1}, {256, 1, 1});
  };
  auto mhc_post = [&](const SharedBuffer& out) {
    enc.set_function(k.mhc_post);
    enc.set_buffer(0, s.S);
    enc.set_buffer(1, out);
    enc.set_buffer(2, s.coef);
    enc.set_constant(3, H);
    enc.set_constant(4, c.streams);
    enc.dispatch({(unsigned)H, (unsigned)n, 1}, {64, 1, 1});
  };

  // PROFILE (VPIPE_MAGI2_DIT_PROFILE=1): a commit and wait at each section
  // boundary, the time between charged to the section. It takes away the
  // overlap a streamed layer's read relies on: read the GPU's split from
  // it, never a streamed forward's wall clock.
  static const bool prof = [] {
    const char* e = std::getenv("VPIPE_MAGI2_DIT_PROFILE");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  enum { kPEmbed, kPMhc, kPProj, kPAttn, kPMlp, kPRoute, kPExpert,
         kPShared, kPN };
  static const char* const kPName[kPN] = {
      "embed", "hyper-connections", "attention projections", "attention",
      "dense MLP", "MoE routing", "routed experts", "MoE shared + merge"};
  double pt[kPN] = {};
  auto t_last = std::chrono::steady_clock::now();
  auto mark = [&](int bucket) {
    if (!prof) { return; }
    enc.end();
    _flops.flush(_stream);
    _stream.commit().wait();
    const auto t = std::chrono::steady_clock::now();
    pt[bucket] += std::chrono::duration<double, std::milli>(t - t_last)
                      .count();
    t_last = t;
    enc = _stream.begin_compute();
  };

  // ---- embedding: a span at a time, in chunks of rows -------------------
  // Every sample's inputs sit at their own offsets in s.in: the whole
  // batch is encoded before anything runs.
  std::vector<SharedBuffer> tembs;
  std::size_t in_off = 0;
  std::size_t in_cap = 0;
  for (int b = 0; b < nb; ++b) {
    const Sample& smp = *batch[(std::size_t)b];
    in_cap += ((std::size_t)smp.T * smp.H * smp.W * c.video_c +
               (std::size_t)smp.audio_len * c.audio_c +
               (std::size_t)(smp.text_len + smp.images.size()) * c.text_c) *
                  4 + 6 * 256;
    // ...and each span (video, every image, audio, text, the specials)
    // rounded up to 256 bytes.
    for (const Image& im : smp.images) {
      in_cap += (std::size_t)im.h * im.w * c.video_c * 4 + 256;
    }
  }
  if (s.in.byte_size() < in_cap) {
    s.in = _mc->make_shared_buffer(in_cap);
    if (s.in.empty()) { return fail("the input rows"); }
  }
  for (int b = 0; b < nb; ++b) {
    const Sample& smp = *batch[(std::size_t)b];
    const Layout& L = Ls[(std::size_t)b];
    enter(b);
    struct Span {
      int kind;        // 0 video 1 audio 2 text
      int row0, rows;
      float t;
      std::vector<float> feat;   // [rows, Cin]
    };
    std::vector<Span> spans;
    const int nvg = smp.T * smp.H * smp.W;
    {
      Span sp{0, 0, nvg, smp.t, {}};
      sp.feat.resize((std::size_t)nvg * c.video_c);
      for (int r = 0; r < nvg; ++r) {
        for (int ch = 0; ch < c.video_c; ++ch) {
          sp.feat[(std::size_t)r * c.video_c + ch] =
              smp.video[(std::size_t)ch * nvg + r];
        }
      }
      spans.push_back(std::move(sp));
    }
    int row = nvg;
    for (const Image& im : smp.images) {
      const int hw = im.h * im.w;
      Span sp{0, row, hw, 0.0f, {}};
      sp.feat.resize((std::size_t)hw * c.video_c);
      for (int r = 0; r < hw; ++r) {
        for (int ch = 0; ch < c.video_c; ++ch) {
          sp.feat[(std::size_t)r * c.video_c + ch] =
              im.latent[(std::size_t)ch * hw + r];
        }
      }
      row += hw;
      spans.push_back(std::move(sp));
    }
    spans.push_back(Span{1, L.nv, smp.audio_len, smp.t,
                         std::vector<float>(smp.audio,
                                            smp.audio + (std::size_t)
                                                smp.audio_len * c.audio_c)});
    spans.push_back(Span{2, L.nv + L.na, smp.text_len, 0.0f,
                         std::vector<float>(smp.text,
                                            smp.text + (std::size_t)
                                                smp.text_len * c.text_c)});
    if (!smp.images.empty()) {
      Span sp{2, L.nv + L.na + smp.text_len, (int)smp.images.size(), 0.0f,
              {}};
      for (const Image& im : smp.images) {
        sp.feat.insert(sp.feat.end(), im.special, im.special + c.text_c);
      }
      spans.push_back(std::move(sp));
    }
    // The inputs, all spans back to back in s.in; the time embeddings.
    for (const Span& sp : spans) {
      const std::vector<float> te = time_embedding(sp.t, c.time_dim);
      SharedBuffer tb = _mc->make_shared_buffer(te.size() * 4);
      std::memcpy(tb.contents(), te.data(), te.size() * 4);
      tembs.push_back(std::move(tb));
      if (!sp.feat.empty()) {
        std::memcpy((char*)s.in.contents() + in_off, sp.feat.data(),
                    sp.feat.size() * 4);
      }
      const int cin = sp.kind == 0 ? c.video_c
                      : sp.kind == 1 ? c.audio_c : c.text_c;
      for (int r0 = 0; r0 < sp.rows; r0 += 4096) {
        const int rows = std::min(4096, sp.rows - r0);
        g32(false, s.in, in_off + (std::size_t)r0 * cin * 4, cin,
            _emb_w[sp.kind], 0, cin, &_emb_b[sp.kind], nullptr, 0, s.emb, 0,
            W4, rows, W4, cin);
        enc.set_function(k.embed_finish);
        enc.set_buffer(0, s.emb);
        enc.set_buffer(1, s.S, (std::size_t)(sp.row0 + r0) * W4 * 2);
        enc.set_buffer(2, tembs.back());
        enc.set_constant(3, W4);
        enc.set_constant(4, c.time_dim);
        enc.dispatch({(unsigned)W4, (unsigned)rows, 1}, {256, 1, 1});
      }
      in_off += sp.feat.size() * 4;
      in_off = (in_off + 255) & ~(std::size_t)255;
    }
    leave(b);
  }
  _flops.flush(_stream);

  mark(kPEmbed);
  // ---- the layers --------------------------------------------------------
  const int E = c.moe_heads * c.moe_experts;
  const int per_tok = c.moe_heads * c.top_k;
  const std::uint64_t per_pick = 3 * gemm_flops(1, c.expert_inter, 256);
  // A RANK's experts: El of each head's, Eloc in all, numbered head * El +
  // e - e0 by the route; every other rank's pick goes to one DROPPED bin
  // (Eloc), sorted with the rest and neither computed nor combined. The
  // whole model: every pick its own, one bin an expert, as before.
  const int El = sh.experts();
  const int Eloc = rank ? c.moe_heads * El : E;
  const int bins = rank ? Eloc + 1 : E;
  const int drop = rank ? Eloc : -1;
  // THE EXCHANGES (a rank): sample b's partials in lane b's buffers. A
  // handoff is encoded between encoders, into the open command buffer:
  // the lane's signal after the dispatch that wrote its partial, a wait
  // before the first that reads the sum (TpOverlap). Overlapped across
  // the batch where its samples have a lane each.
  // VPIPE_MAGI2_TP_LAG=0 runs them in line (the same arithmetic; read
  // per forward, so one process can run both).
  const bool lag_env = [] {
    const char* e = std::getenv("VPIPE_MAGI2_TP_LAG");
    return !(e != nullptr && std::atoi(e) == 0);
  }();
  const bool lag = rank && lag_env && nb >= 2 && nb <= kTpLanes;
  _tp_lagged = lag;
  if (rank) {
    const std::size_t pb = (std::size_t)s.n * (std::size_t)H * 2;
    for (int l = 0; l < (lag ? nb : 1); ++l) {
      if (!tp_buffer_(_tp_part[l], pb) || !tp_buffer_(_tp_recv[l], pb)) {
        enc.end();
        _stream.commit().wait();
        join_reads();
        return fail("tensor parallel: no room for the partial sums");
      }
    }
  }
  auto tp_post = [&](int lane) {
    const TpOverlap::Ticket t =
        _tp_pipe->post(lane, _tp_part[lane], _tp_recv[lane],
                       (std::size_t)n * (std::size_t)H * 2, _tp_seq++);
    ++_tp_exchanges;
    enc.end();
    _stream.encode_signal(*t.ready, t.ready_at);
    enc = _stream.begin_compute();
    return t.done_at;
  };
  // The others' sum awaited, added to this rank's: s.P then holds the
  // sublayer's output whole, as one model's would.
  auto tp_sum = [&](int lane, std::uint64_t done_at) {
    enc.end();
    _stream.encode_wait(_tp_pipe->done(), done_at);
    enc = _stream.begin_compute();
    add(_tp_part[lane], _tp_recv[lane], s.P, (std::size_t)n * H);
  };
  if (_keep_routes) { _kept.clear(); }
  int layer_pos = -1;
  for (const Layer& ly : _layers) {
    ++layer_pos;
    // The experts: held, or this layer's slot once its read is in -- one
    // read for the whole batch.
    const SharedBuffer* ex = nullptr;
    if (ly.moe && ly.streamed) {
      const int slot = streamed_seen % 2;
      if (!reads[slot].get()) {
        join_reads();
        enc.end();
        _stream.commit().wait();
        return fail(fmt("layer {}: the experts' read failed", ly.idx)());
      }
      ex = _slot[slot];
    } else if (ly.moe) {
      ex = ly.wx;
    }
    // A LAYER'S TWO SUBLAYERS, each its output into `out`: the attention
    // and the MLP of the sample in the scratch (enter / leave), each
    // after its hyper-connection's pre step. Whole, `out` is the
    // sublayer's output; on a rank, its PARTIAL sum, which the others'
    // complete before the hyper-connection's post step takes it.
    std::string perr;
    auto attn_part = [&](const SharedBuffer& out) -> bool {
      // Attention.
      mhc_pre(ly, ly.phi_attn, ly.ab_attn);
      mark(kPMhc);
      rms_rows(ly.m, s.Y, s.A, ly.attn_norm, H);
      mm_proj(ly.m, s.A, ly.qkv, s.qkv);
      // The per-head gate stays bf16: 24 outputs a row.
      for (int i = 0; i < (ly.m == 1 ? 1 : 3); ++i) {
        const int r0 = ly.m == 1 ? 0 : span0[i];
        const int rn = ly.m == 1 ? n : spann[i];
        gm.dense(enc, s.A, (std::size_t)r0 * H * 2, ly.g,
                 (std::size_t)(ly.m == 1 ? 0 : i) * nq * H * 2, s.g,
                 (std::size_t)r0 * nq * 2, rn, nq, H);
      }
      mark(kPProj);
      for (int i = 0; i < 3; ++i) {
        const int r0 = ly.m == 1 ? (i == 0 ? 0 : n) : span0[i];
        const int rn = ly.m == 1 ? (i == 0 ? n : 0) : spann[i];
        if (rn <= 0) { continue; }
        const std::size_t wo =
            (std::size_t)(ly.m == 1 ? 0 : i) * c.head_dim * 4;
        enc.set_function(k.qk_rope);
        enc.set_buffer(0, s.qkv);
        enc.set_buffer(1, s.sin);
        enc.set_buffer(2, s.cos);
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
      }
      {
        // The matrix-core kernel's tiles (64 / 32) on an M5, the steel one's
        // (32 / 16) else; the sinks either way.
        const int BQ = _nax ? 64 : 32, BK = _nax ? 32 : 16;
        constexpr int HD = 128;
        SteelAttnParams pp{};
        pp.B = 1; pp.H = nq; pp.D = HD;
        pp.qL = n; pp.kL = n;
        pp.gqa_factor = nq / nkv;
        pp.scale = 1.0f / std::sqrt((float)HD);
        pp.NQ = (n + BQ - 1) / BQ;
        pp.NK = (n + BK - 1) / BK;
        pp.NQ_aligned = n / BQ;
        pp.NK_aligned = n / BK;
        pp.qL_rem = n - pp.NQ_aligned * BQ;
        pp.kL_rem = n - pp.NK_aligned * BK;
        pp.qL_off = 0;
        for (std::int64_t* st : {pp.Q_strides, pp.K_strides, pp.V_strides}) {
          st[0] = (std::int64_t)n * ldqkv; st[1] = HD; st[2] = ldqkv;
        }
        pp.O_strides[0] = (std::int64_t)n * qs;
        pp.O_strides[1] = HD;
        pp.O_strides[2] = qs;
        const int al = (n % BQ == 0 ? 1 : 0) | (n % BK == 0 ? 2 : 0);
        // SageAttention: Q and K (and V) quantized into its scratch, then
        // the attention's int8 twin over them.
        const bool sage = _sage != nullptr && layer_pos >= c.sage.dense_layers;
        const bool pv8 = sage && pv8_step;
        if (sage) {
          using Op = MetalSageAttention::Operand;
          const Op qo{&s.qkv, 0, ldqkv, HD};
          const Op ko{&s.qkv, (std::size_t)qs, ldqkv, HD};
          const Op vo{&s.qkv, (std::size_t)(qs + kvs), ldqkv, HD};
          sage::Config sc = c.sage;
          sc.pv8 = pv8;
          std::string se;
          if (!_sage->prepare(enc, qo, ko, nq, nkv, n, n, HD, BQ, BK,
                              sc, &se, pv8 ? &vo : nullptr)) {
            perr = "SageAttention: " + se;
            return false;
          }
          ++_n_sage;
          if (pv8) { ++_n_sage_pv8; }
        }
        enc.set_function(sage   ? k.nax_sage[al][pv8 ? 1 : 0]
                         : _nax ? k.nax_fn[al]
                                : k.attn_fn[al]);
        enc.set_buffer(0, s.qkv);
        enc.set_buffer(1, s.qkv, (std::size_t)qs * 2);
        enc.set_buffer(2, s.qkv, (std::size_t)(qs + kvs) * 2);
        enc.set_buffer(3, s.O);
        enc.set_constant_bytes(4, &pp, sizeof pp);
        enc.set_buffer(7, ly.sinks);
        if (sage) { _sage->bind(enc); }
        enc.dispatch({32u * (unsigned)pp.NQ, 4u * (unsigned)nq, 1},
                     {32, 4, 1});
        // Every query sees every key (one sample, no mask); the sink is
        // a column of the softmax, not a key.
        _flops.attention(nq, (long long)n * n, HD);
      }
      enc.set_function(k.gate);
      enc.set_buffer(0, s.O);
      enc.set_buffer(1, s.g);
      enc.set_constant(2, nq);
      enc.set_constant(3, c.head_dim);
      enc.set_constant(4, nq);
      enc.set_constant(5, 0);
      enc.dispatch({(unsigned)qs, (unsigned)n, 1}, {128, 1, 1});
      mark(kPAttn);
      mm_proj(ly.m, s.O, ly.proj, out);
      mark(kPProj);
      return true;
    };
    auto mlp_part = [&](const SharedBuffer& out) -> bool {
      // MLP.
      mhc_pre(ly, ly.phi_mlp, ly.ab_mlp);
      mark(kPMhc);
      if (!ly.moe) {
        rms_rows(ly.m, s.Y, s.A, ly.mlp_norm, H);
        mm_proj(ly.m, s.A, ly.up, s.U);
        swiglu(s.U, s.act, sh.mlp(), n);
        mm_proj(ly.m, s.act, ly.down, out);
        mark(kPMlp);
        return true;
      }
      rms_rows(3, s.Y, s.A, ly.mlp_norm, H);
      gm.proj(enc, s.A, 0, ly.split[0], s.split, 0, n);
      for (int t0 = 0; t0 < n; t0 += s.chunk) {
        const int tn = std::min(s.chunk, n - t0);
        const int np = tn * per_tok;
        const int maxtiles = (np + kMaxm - 1) / kMaxm + bins;
        const int npad = maxtiles * kMaxm;
        // Router logits per head: [tn, 256] x gate_h [E_h, 256]^T, f32.
        g32(true, s.split, (std::size_t)t0 * H * 2, H, ly.rgate, 0, 256,
            nullptr, nullptr, 0, s.logits, 0, E, tn, c.moe_experts, 256,
            c.moe_heads, 256, (long)c.moe_experts * 256, c.moe_experts);
        enc.set_function(k.topk);
        enc.set_buffer(0, s.logits);
        enc.set_buffer(1, ly.rbias);
        enc.set_buffer(2, s.eid);
        enc.set_buffer(3, s.prob);
        enc.set_constant(4, c.moe_heads);
        enc.set_constant(5, c.moe_experts);
        enc.set_constant(6, c.top_k);
        enc.set_constant(7, c.route_scale);
        enc.set_constant(8, E);
        {
          const SharedBuffer* fb = nullptr;
          for (const auto& f : _forced) {
            if (f.first == ly.idx) { fb = &f.second; }
          }
          enc.set_buffer(9, fb != nullptr ? *fb : s.eid,
                         fb != nullptr
                             ? (std::size_t)t0 * per_tok * 4 : 0);
          enc.set_constant(10, fb != nullptr ? 1 : 0);
        }
        enc.set_constant(11, sh.e0);
        enc.set_constant(12, sh.e1);
        enc.dispatch({(unsigned)(32 * c.moe_heads), (unsigned)tn, 1},
                     {32, 1, 1});
        if (_keep_routes) {
          // Tests: drain, then read this chunk's picks back.
          enc.end();
          _flops.flush(_stream);
          _stream.commit().wait();
          const int* e = (const int*)s.eid.contents();
          std::vector<int>* dst = nullptr;
          for (auto& kp : _kept) {
            if (kp.first == ly.idx) { dst = &kp.second; }
          }
          if (dst == nullptr) {
            _kept.emplace_back(ly.idx, std::vector<int>());
            dst = &_kept.back().second;
          }
          for (int i = 0; i < np; ++i) {
            // A rank's own picks back to the head's experts; the others'
            // (its dropped bin) unknown here.
            dst->push_back(e[i] == drop ? -1 : e[i] % El + sh.e0);
          }
          enc = _stream.begin_compute();
        }
        auto fill = [&](const SharedBuffer& b, int v, int cnt) {
          enc.set_function(k.ifill);
          enc.set_buffer(0, b);
          enc.set_constant(1, v);
          enc.set_constant(2, cnt);
          enc.dispatch({(unsigned)cnt, 1, 1}, {256, 1, 1});
        };
        fill(s.hist, 0, bins);
        fill(s.tile2e, -1, maxtiles);
        fill(s.srow, -1, npad);
        fill(s.sdst, -1, npad);
        enc.set_function(k.hist);
        enc.set_buffer(0, s.eid);
        enc.set_buffer(1, s.hist);
        enc.set_constant(2, np);
        enc.dispatch({(unsigned)np, 1, 1}, {256, 1, 1});
        enc.set_function(k.sort_setup);
        enc.set_buffer(0, s.hist);
        enc.set_buffer(1, s.boff);
        enc.set_buffer(2, s.tile2e);
        enc.set_buffer(3, s.cursor);
        enc.set_buffer(4, s.nt);
        enc.set_constant(5, bins);
        enc.set_constant(6, kMaxm);
        enc.set_constant(7, maxtiles);
        enc.dispatch({1, 1, 1}, {1, 1, 1});
        if (drop >= 0) {
          // The other ranks' picks sorted into the dropped bin: its tiles
          // off the expert kernels' grid.
          enc.set_function(k.drop_bin);
          enc.set_buffer(0, s.tile2e);
          enc.set_constant(1, maxtiles);
          enc.set_constant(2, drop);
          enc.dispatch({(unsigned)maxtiles, 1, 1}, {256, 1, 1});
        }
        enc.set_function(k.scatter);
        enc.set_buffer(0, s.eid);
        enc.set_buffer(1, s.boff);
        enc.set_buffer(2, s.cursor);
        enc.set_buffer(3, s.srow);
        enc.set_buffer(4, s.sdst);
        enc.set_constant(5, np);
        enc.set_constant(6, c.top_k);
        enc.dispatch({(unsigned)np, 1, 1}, {256, 1, 1});
        enc.set_function(k.invert);
        enc.set_buffer(0, s.sdst);
        enc.set_buffer(1, s.pos);
        enc.set_constant(2, npad);
        enc.dispatch({(unsigned)npad, 1, 1}, {256, 1, 1});
        if (_nax_moe) {
          // Each expert's sorted 32-row tiles two at a time (its last item
          // what is left), so a threadgroup runs up to 64 rows of one expert.
          enc.set_function(k.tile_items);
          enc.set_buffer(0, s.hist);
          enc.set_buffer(1, s.boff);
          enc.set_buffer(2, s.items);
          enc.set_constant(3, Eloc);
          enc.set_constant(4, kMaxm);
          enc.set_constant(5, maxtiles);
          enc.set_constant(6, 2);
          enc.dispatch({1, 1, 1}, {1, 1, 1});
        }
        mark(kPRoute);
        enc.set_function(_nax_moe  ? k.expert_nax
                         : _tg_moe ? k.expert_tg
                                   : k.expert);
        enc.set_buffer(0, s.split, (std::size_t)t0 * H * 2);
        enc.set_buffer(1, s.srow);
        enc.set_buffer(2, s.sdst);
        enc.set_buffer(3, s.prob);
        enc.set_buffer(4, s.tile2e);
        enc.set_buffer(5, ex[0]);
        enc.set_buffer(6, ex[1]);
        enc.set_buffer(7, ex[2]);
        enc.set_buffer(8, s.yexp);
        enc.set_constant(9, c.expert_inter);
        if (_nax_moe) {
          enc.set_buffer(10, s.items);
          enc.dispatch({(unsigned)(256 * maxtiles), 1, 1}, {256, 1, 1});
          ++_n_moe_nax;
        } else if (_tg_moe) {
          enc.dispatch({(unsigned)(256 * maxtiles), 1, 1}, {256, 1, 1});
        } else {
          enc.dispatch({(unsigned)(128 * maxtiles), 1, 1}, {128, 1, 1});
        }
        // Each pick: gate and up [256 -> I], down [I -> 256]. A rank's
        // own picks are sorted out on the GPU; it counts its share.
        _flops.add((std::uint64_t)np * per_pick * (std::uint64_t)El /
                   (std::uint64_t)c.moe_experts);
        mark(kPExpert);
        enc.set_function(k.combine);
        enc.set_buffer(0, s.yexp);
        enc.set_buffer(1, s.pos);
        enc.set_buffer(2, s.comb, (std::size_t)t0 * H * 2);
        enc.set_constant(3, c.top_k);
        enc.set_buffer(4, s.eid);
        enc.set_constant(5, drop);
        enc.dispatch({256, (unsigned)(tn * c.moe_heads), 1}, {256, 1, 1});
        mark(kPRoute);
      }
      gm.proj(enc, s.comb, 0, ly.merge[0], s.merge, 0, n);
      gm.proj(enc, s.A, 0, ly.sh1[0], s.x1, 0, n);
      mm_proj(3, s.A, ly.ms1, s.x2);
      swiglu(s.x1, s.a1, sh.shared(), n);
      swiglu(s.x2, s.a2, sh.ms(), n);
      gm.proj(enc, s.a1, 0, ly.sh2[0], s.y1, 0, n);
      mm_proj(3, s.a2, ly.ms2, s.y2);
      add(s.y1, s.y2, s.y1, (std::size_t)n * H);
      add(s.merge, s.y1, out, (std::size_t)n * H);
      mark(kPShared);
      return true;
    };
    auto part_failed = [&](int b) {
      enc.end();
      _stream.commit().wait();
      leave(b);
      join_reads();
      return fail(perr);
    };
    if (!rank) {
      for (int b = 0; b < nb; ++b) {
        enter(b);
        if (!attn_part(s.P)) { return part_failed(b); }
        mhc_post(s.P);
        if (!mlp_part(s.P)) { return part_failed(b); }
        mhc_post(s.P);
        mark(kPMhc);
        leave(b);
      }
    } else if (!lag) {
      // A RANK, in line: each partial posted, its sum awaited at once.
      for (int b = 0; b < nb; ++b) {
        enter(b);
        if (!attn_part(_tp_part[0])) { return part_failed(b); }
        tp_sum(0, tp_post(0));
        mhc_post(s.P);
        if (!mlp_part(_tp_part[0])) { return part_failed(b); }
        tp_sum(0, tp_post(0));
        mhc_post(s.P);
        mark(kPMhc);
        leave(b);
      }
    } else {
      // A RANK, OVERLAPPED across the batch: every sample's attention, its
      // partial posted; then each sample's sum, hyper-connection and MLP,
      // its partial posted; then each MLP's sum. A sample's exchange
      // crosses under the next sample's work, the attention's under the
      // first MLP's too; the last MLP's alone is exposed, once a layer.
      // Lane b is sample b's, each write after the GPU waited for the
      // exchange of the lane's last partial; the waits in the order the
      // partials were posted, which is the order the GPU signals them.
      std::uint64_t va[kTpLanes], vm[kTpLanes];
      for (int b = 0; b < nb; ++b) {
        enter(b);
        if (!attn_part(_tp_part[b])) { return part_failed(b); }
        va[b] = tp_post(b);
        leave(b);
      }
      for (int b = 0; b < nb; ++b) {
        enter(b);
        tp_sum(b, va[b]);
        mhc_post(s.P);
        if (!mlp_part(_tp_part[b])) { return part_failed(b); }
        vm[b] = tp_post(b);
        leave(b);
      }
      for (int b = 0; b < nb; ++b) {
        enter(b);
        tp_sum(b, vm[b]);
        mhc_post(s.P);
        mark(kPMhc);
        leave(b);
      }
    }
    // The layer's count onto the buffer it ends in. A held stack commits
    // nothing between layers; flushed a layer at a time, it is credited
    // as the stream's auto-split buffers retire, not all at the end.
    _flops.flush(_stream);
    if (ly.streamed) {
      // This layer's slot is free again once its work is done: commit,
      // issue the read two streamed layers on into it under this layer's
      // GPU work, wait, and carry on in a fresh command buffer.
      enc.end();
      auto fence = _stream.commit();
      ++streamed_seen;
      std::string ge;
      const bool ok = fence.wait_ok(&ge);
      if (next_read < (int)streamed.size()) { read_into(next_read++); }
      if (!ok) {
        join_reads();
        return fail(fmt("layer {}: {}", ly.idx, ge)());
      }
      if (_block_cb && !_block_cb(layer_pos + 1, (int)_layers.size())) {
        join_reads();
        if (err != nullptr) { err->clear(); }
        return false;
      }
      enc = _stream.begin_compute();
    }
  }

  // ---- the heads, a sample at a time (s.vel holds one) -----------------
  vel_video->assign((std::size_t)nb, {});
  vel_audio->assign((std::size_t)nb, {});
  for (int b = 0; b < nb; ++b) {
  const Sample& smp = *batch[(std::size_t)b];
  const Layout& L = Ls[(std::size_t)b];
  enter(b);
  const int nvg = smp.T * smp.H * smp.W;
  enc.set_function(k.inv_rms);
  enc.set_buffer(0, s.S);
  enc.set_buffer(1, s.rs);
  enc.set_constant(2, W4);
  enc.set_constant(3, c.eps);
  enc.dispatch({(unsigned)(256 * n), 1, 1}, {256, 1, 1});
  g32(true, s.S, 0, W4, _out_w[0], 0, W4, nullptr, &s.rs, 0, s.vel, 0,
      c.video_c, nvg, c.video_c, W4);
  const std::size_t aoff = (std::size_t)nvg * c.video_c * 4;
  g32(true, s.S, (std::size_t)L.nv * W4 * 2, W4, _out_w[1], 0, W4, nullptr,
      &s.rs, (std::size_t)L.nv * 4, s.vel, aoff, c.audio_c, smp.audio_len,
      c.audio_c, W4);
  enc.end();
  _flops.flush(_stream);
  std::string ge;
  const bool done = _stream.commit().wait_ok(&ge);
  leave(b);
  if (!done) {
    join_reads();
    return fail("the forward: " + ge);
  }
  const float* v = (const float*)s.vel.contents();
  std::vector<float>& ov = (*vel_video)[(std::size_t)b];
  ov.assign((std::size_t)c.video_c * nvg, 0.0f);
  for (int r = 0; r < nvg; ++r) {
    for (int ch = 0; ch < c.video_c; ++ch) {
      ov[(std::size_t)ch * nvg + r] = v[(std::size_t)r * c.video_c + ch];
    }
  }
  const float* va = (const float*)((const char*)s.vel.contents() + aoff);
  (*vel_audio)[(std::size_t)b].assign(
      va, va + (std::size_t)smp.audio_len * c.audio_c);
  if (b + 1 < nb) { enc = _stream.begin_compute(); }
  }
  join_reads();
  if (prof) {
    std::string line = "[magi2] preview DiT profile (ms):";
    double all = 0.0;
    for (int i = 0; i < kPN; ++i) { all += pt[i]; }
    for (int i = 0; i < kPN; ++i) {
      line += fmt(" {} {:.0f} ({:.0f}%);", kPName[i], pt[i],
                  all > 0.0 ? 100.0 * pt[i] / all : 0.0)();
    }
    std::fprintf(stderr, "%s\n", line.c_str());
  }
  if (_block_cb && !_block_cb((int)_layers.size(), (int)_layers.size())) {
    if (err != nullptr) { err->clear(); }
    return false;
  }
  return true;
}

}  // namespace genai
}  // namespace vpipe
