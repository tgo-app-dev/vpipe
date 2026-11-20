#include "generative-models/relayvsr/metal-relayvsr-decoder.h"

#include "common/flex-bag.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/relayvsr/relayvsr-ops.h"
#include "generative-models/shared/ane-ffn.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace vpipe {
namespace genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

constexpr const char* kKey = "relayvsr-decoder/";

}  // namespace

struct MetalRelayVsrDecoder::Block {
  relayvsr::Proj q, k, v, ck, cv, o, gu, dn;
  SharedBuffer an, kn, vn, fn;
};

struct MetalRelayVsrDecoder::Scratch {
  int ph = 0, pw = 0, h = 0, w = 0, n = 0;
  SharedBuffer e0, e1, e2, x, y, q, a, o, gu, sw, hd, res, kr_cur, kr_prev,
      cr;
  SharedBuffer hk[2], hv[2];          // [depth][n, kv] the history
  SharedBuffer tab_t0, tab_tm1, tab_h, tab_w;
  int cur = 0;                         // which of hk / hv this frame fills
};

MetalRelayVsrDecoder::~MetalRelayVsrDecoder()
{
  if (_ane) { _ane->join(); }
}

bool
MetalRelayVsrDecoder::uses_matrix_cores() const noexcept
{
  return _ops && _ops->matrix_cores();
}

bool
MetalRelayVsrDecoder::window_on_matrix_cores() const noexcept
{
  return _ops && _ops->nax_attention();
}

bool
MetalRelayVsrDecoder::config_from_checkpoint(const std::string& flash_file,
                                             Config* out, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  auto w = MetalLlamaWeights::open(flash_file);
  if (!w) {
    return fail(fmt("'{}' is not a safetensors file", flash_file)());
  }
  const auto& md = w->metadata();
  auto it = md.find("format");
  // The x2 file (lightx2v's, and ComfyUI's "flash VAE x2") states its scale
  // and leaves the S variant's defaults out of its config.
  const bool x2 = it != md.end() && it->second == "relayvsr-flash-x2-v1";
  if (it == md.end() || (it->second != "refvsr-flash-v1" && !x2)) {
    return fail(fmt("'{}' is not a RelayVSR decoder checkpoint (format "
                    "refvsr-flash-v1, or relayvsr-flash-x2-v1)",
                    flash_file)());
  }
  auto ct = md.find("config");
  if (ct == md.end()) { return fail("refvsr-flash: no config in the file"); }
  const FlexData c = FlexData::from_json(ct->second);
  // The reference's variants (FlashVSRConditionNet.VARIANTS).
  struct V { const char* name; int depth, dim, heads, kv; };
  static const V kVariants[] = {
      {"halfS", 7, 512, 8, 2}, {"S", 10, 512, 8, 2}, {"M", 11, 640, 10, 2},
      {"B", 12, 768, 12, 3},   {"L", 14, 1024, 16, 4},
      {"XL", 14, 1536, 24, 3}};
  const std::string variant = bag::text(c, "variant", "S");
  const V* v = nullptr;
  for (const V& k : kVariants) {
    std::string a(k.name), b(variant);
    for (char& ch : a) { ch = (char)std::toupper((unsigned char)ch); }
    for (char& ch : b) { ch = (char)std::toupper((unsigned char)ch); }
    if (a == b) { v = &k; }
  }
  if (v == nullptr) {
    return fail(fmt("refvsr-flash: unknown variant '{}'", variant)());
  }
  Config cfg;
  cfg.depth = (int)bag::integer(c, "depth", v->depth);
  cfg.dim = (int)bag::integer(c, "dim", v->dim);
  cfg.heads = (int)bag::integer(c, "num_heads", v->heads);
  cfg.kv_heads = (int)bag::integer(c, "num_kv_groups", v->kv);
  cfg.head_dim = cfg.dim / cfg.heads;
  const double ratio = bag::real(c, "mlp_ratio", 8.0 / 3.0);
  cfg.ffn = (int)std::ceil((double)cfg.dim * ratio / 64.0) * 64;
  cfg.cond_channels = (int)bag::integer(c, "condition_channels", 48);
  cfg.head_width = (int)bag::integer(c, "spatial_head_dim", 1024);
  cfg.history = (int)bag::integer(c, "window_size", 3);
  cfg.eps = (float)bag::real(c, "norm_eps", 1e-5);
  cfg.upscale = (int)bag::integer(c, "upscale_factor", 4);
  const FlexData enc = bag::value(c, "encoder_channels");
  if (enc.is_array() && enc.as_array().size() == 3) {
    for (int i = 0; i < 3; ++i) {
      cfg.enc[i] = (int)enc.as_array()[(std::size_t)i].as_int();
    }
  }
  const FlexData sa = bag::value(c, "spatial_attention");
  auto win = [&](const char* key, int* wv) {
    const FlexData a = bag::value(sa, key);
    if (a.is_array() && a.as_array().size() == 2) {
      wv[0] = (int)a.as_array()[0].as_int();
      wv[1] = (int)a.as_array()[1].as_int();
    }
  };
  win("video_window_size", cfg.win_video);
  win("start_condition_window_size", cfg.win_start);
  win("end_condition_window_size", cfg.win_end);
  win("condition_self_window_size", cfg.win_self);
  std::string why;
  if (bag::text(c, "condition_mode", "start") != "endpoints") {
    why = "condition_mode is not 'endpoints'";
  } else if (bag::text(c, "temporal_rope_policy", "current_relative") !=
             "current_relative") {
    why = "temporal_rope_policy is not current_relative";
  } else if (cfg.upscale != (x2 ? 2 : 4)) {
    why = fmt("upscale_factor {} in a {} file", cfg.upscale, it->second)();
  } else if (!sa.is_object()) {
    why = "it has no spatial_attention (dense attention is not ported)";
  } else if (bag::text(sa, "type", "query_centered") != "query_centered" ||
             bag::text(sa, "interval", "half_open") != "half_open" ||
             bag::text(sa, "boundary", "truncated") != "truncated") {
    why = "its windows are not query-centred, half-open and truncated";
  } else if (cfg.head_dim != 64) {
    why = fmt("head dim {} (the 3-D RoPE is 16 + 24 + 24)", cfg.head_dim)();
  } else if (cfg.history < 1 || cfg.history > 2) {
    why = fmt("window_size {} (1 or 2 frames of history ported)",
              cfg.history)();
  } else if (cfg.dim % 64 != 0 || cfg.heads % cfg.kv_heads != 0) {
    why = "an unsupported width or head grouping";
  }
  for (const int* wv : {cfg.win_video, cfg.win_start, cfg.win_end,
                        cfg.win_self}) {
    if (why.empty() && (wv[0] <= 0 || wv[1] <= 0 || wv[0] % 2 != 0 ||
                        wv[1] % 2 != 0)) {
      why = "a spatial window that is not two positive even sizes";
    }
  }
  if (!why.empty()) {
    return fail(fmt("refvsr-flash: '{}' -- {}; this port runs the "
                    "release's configuration only", flash_file, why)());
  }
  *out = cfg;
  return true;
}

std::uint64_t
MetalRelayVsrDecoder::condition_bytes(const Config& cfg, int ph, int pw)
{
  const int ls = cfg.latent_stride();
  const std::uint64_t n = (std::uint64_t)(ph / ls) * (pw / ls);
  return 2u * (std::uint64_t)cfg.depth * n *
         (std::uint64_t)(cfg.kv_heads * cfg.head_dim) * 2u;
}

std::uint64_t
MetalRelayVsrDecoder::scratch_bytes(const Config& cfg, int ph, int pw)
{
  // The encoder at 1/2, 1/4 and the latent grid (1/4 at 4x, 1/8 at 2x).
  const int ls = cfg.latent_stride();
  const std::uint64_t n = (std::uint64_t)(ph / ls) * (pw / ls);
  const std::uint64_t n2 = (std::uint64_t)(ph / 2) * (pw / 2);
  const std::uint64_t n4 = (std::uint64_t)(ph / 4) * (pw / 4);
  const std::uint64_t kv = (std::uint64_t)(cfg.kv_heads * cfg.head_dim);
  const std::uint64_t D = (std::uint64_t)cfg.dim;
  std::uint64_t b = 0;
  b += n2 * (std::uint64_t)cfg.enc[0] * 2u;
  b += n4 * (std::uint64_t)cfg.enc[1] * 2u;
  b += n * (std::uint64_t)cfg.enc[2] * 2u;
  b += 5u * n * D * 2u;                                // x y q a o
  b += n * (std::uint64_t)(3 * cfg.ffn) * 2u;          // gu, sw
  b += n * (std::uint64_t)(cfg.head_width + 768) * 2u; // hd, res
  b += 2u * n * kv * 2u;                               // kr_cur, kr_prev
  b += n * (std::uint64_t)cfg.cond_channels * 2u;      // cr
  b += 4u * (std::uint64_t)cfg.depth * n * kv * 2u;    // the history
  // The int8 context's over the blocks' projections: q and o [D, D], the
  // four K / V ones [KV, D], gate|up [2F, D], down [D, F]. The down is the
  // one the accurate route takes -- on tier 0, and on fast and token too,
  // whose layouts cannot take its 1408 -- with its per-call weight and,
  // where it tunes, two split planes.
  const int Di = cfg.dim, KVi = (int)kv, Fi = cfg.ffn;
  b += relayvsr::Ops::i8_scratch_bytes(
      cfg.i8_tier, cfg.autotune, (int)n,
      {{Di, Di}, {KVi, Di}, {2 * Fi, Di}, {Di, Fi}});
  // SageAttention's int8 operands: a frame's four planes.
  b += relayvsr::Ops::window_int8_bytes(4, cfg.heads, cfg.kv_heads, (int)n,
                                        cfg.head_dim, cfg.sage_mode());
  return b;
}

std::uint64_t
MetalRelayVsrDecoder::weight_bytes(const Config& cfg)
{
  const std::uint64_t D = (std::uint64_t)cfg.dim;
  const std::uint64_t kv = (std::uint64_t)(cfg.kv_heads * cfg.head_dim);
  const int t = cfg.i8_tier;
  const int Di = cfg.dim, KVi = cfg.kv_heads * cfg.head_dim, Fi = cfg.ffn;
  // A block, in bytes: its projections at the tier's layout, their biases,
  // the norms.
  const std::uint64_t blk =
      2u * relayvsr::proj_bytes(Di, Di, t) +
      4u * relayvsr::proj_bytes(KVi, Di, t) +
      relayvsr::proj_bytes(2 * Fi, Di, t) +
      relayvsr::proj_bytes(Di, Fi, t) +
      (2u * D + 4u * kv + 2u * (std::uint64_t)cfg.head_dim + 2u * D) * 2u;
  const std::uint64_t e0 = (std::uint64_t)cfg.enc[0];
  const std::uint64_t e1 = (std::uint64_t)cfg.enc[1];
  const std::uint64_t e2 = (std::uint64_t)cfg.enc[2];
  const std::uint64_t hw = (std::uint64_t)cfg.head_width;
  const std::uint64_t top =
      27u * e0 + e0 + 9u * e0 * e1 + e1 + 9u * e1 * e2 + e2 + e2 * D + D +
      4u * D + (std::uint64_t)cfg.cond_channels * D + D + hw * D + hw +
      768u * hw + 768u;
  return (std::uint64_t)cfg.depth * blk + top * 2u;
}

std::uint64_t
MetalRelayVsrDecoder::held_bytes() const
{
  std::uint64_t b = weight_bytes(_cfg);
  for (const Block& k : _blocks) {
    for (const relayvsr::Proj* p :
         {&k.q, &k.k, &k.v, &k.ck, &k.cv, &k.o, &k.gu, &k.dn}) {
      b += p->weight_bytes();
      b -= relayvsr::proj_bytes(p->n, p->k, _cfg.i8_tier);
    }
  }
  return b;
}

const relayvsr::Ops&
MetalRelayVsrDecoder::ops() const noexcept
{
  return *_ops;
}

bool
MetalRelayVsrDecoder::ane_stages(const Config& cfg)
{
  // gate|up is dim deep, the down ffn deep.
  return relayvsr::held_group(cfg.dim, cfg.i8_tier) == 0 &&
         relayvsr::held_group(cfg.ffn, cfg.i8_tier) == 0;
}

std::uint64_t
MetalRelayVsrDecoder::ane_runtime_bytes(const Config& cfg)
{
  return AneFeedForward::runtime_bytes(cfg.dim, cfg.ffn, 0,
                                       relayvsr::ane_chunk(1 << 30));
}

std::uint64_t
MetalRelayVsrDecoder::ane_held_bytes() const
{
  return _ane ? AneFeedForward::runtime_bytes(_cfg.dim, _cfg.ffn, 0,
                                              _ane->chunk())
              : 0;
}

std::unique_ptr<MetalRelayVsrDecoder>
MetalRelayVsrDecoder::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                           const Config& cfg, std::string* err)
{
  auto fail = [&](const std::string& m)
      -> std::unique_ptr<MetalRelayVsrDecoder> {
    if (err != nullptr) { *err = m; }
    return nullptr;
  };
  if (!ws || mc == nullptr) { return fail("no weights or backend"); }
  auto m = std::unique_ptr<MetalRelayVsrDecoder>(new MetalRelayVsrDecoder());
  m->_cfg = cfg;
  m->_mc = mc;
  m->_ws = std::move(ws);
  m->_ops = std::make_unique<relayvsr::Ops>();
  if (!m->_ops->load(mc, /*bf16=*/false, err)) { return nullptr; }
  // The int8 tier before the blocks load (a holding tier converts each
  // projection as it does); what it settles on is the tier recorded.
  (void)m->_ops->set_i8(cfg.i8_tier, cfg.autotune);
  m->_cfg.i8_tier = m->_ops->i8_tier();
  // The windows' int8 prologue at a frame's four planes.
  m->_ops->set_window_planes(4);
  WeightSet& w = *m->_ws;
  auto t = [&](const std::string& nm) {
    return relayvsr::tensor_elt(w, mc, nm, false, kKey);
  };
  // A 3x3 conv's weight, tap-major for the gather conv: [Cout, 9 * Cin].
  auto conv = [&](const std::string& nm) {
    return w.derived(std::string(kKey) + "f16/tap-major/" + nm, [&]() {
      relayvsr::RawTensor r;
      if (!relayvsr::read_raw(w, mc, nm, &r)) { return SharedBuffer{}; }
      return relayvsr::tap_major_elt(mc, r, false);
    });
  };
  m->_e0w = conv("lr_encoder.layers.0.weight");
  m->_e0b = t("lr_encoder.layers.0.bias");
  m->_e1w = conv("lr_encoder.layers.2.weight");
  m->_e1b = t("lr_encoder.layers.2.bias");
  m->_e2w = conv("lr_encoder.layers.4.weight");
  m->_e2b = t("lr_encoder.layers.4.bias");
  m->_e3w = t("lr_encoder.layers.6.weight");      // [512, 128, 1, 1]
  m->_e3b = t("lr_encoder.layers.6.bias");
  m->_enorm = t("lr_encoder.norm.weight");
  m->_vtype = t("video_type_embedding");
  m->_cpw = t("condition_encoder.projection.weight");   // [512, 48, 1, 1]
  m->_cpb = t("condition_encoder.projection.bias");
  m->_cnorm = t("condition_encoder.norm.weight");
  m->_ctype = t("condition_type_embedding");
  m->_hw = t("spatial_head.0.weight");
  m->_hb = t("spatial_head.0.bias");
  m->_rw = t("residual_projection.weight");
  m->_rb = t("residual_projection.bias");
  for (const SharedBuffer* b :
       {&m->_e0w, &m->_e0b, &m->_e1w, &m->_e1b, &m->_e2w, &m->_e2b,
        &m->_e3w, &m->_e3b, &m->_enorm, &m->_vtype, &m->_cpw, &m->_cpb,
        &m->_cnorm, &m->_ctype, &m->_hw, &m->_hb, &m->_rw, &m->_rb}) {
    if (b->empty()) {
      return fail("refvsr-flash: a top-level tensor is missing");
    }
  }
  m->_blocks.resize((std::size_t)cfg.depth);
  relayvsr::Ops& op = *m->_ops;
  const int D = cfg.dim, KV = cfg.kv_heads * cfg.head_dim, F = cfg.ffn;
  for (int i = 0; i < cfg.depth; ++i) {
    Block& b = m->_blocks[(std::size_t)i];
    const std::string p = fmt("backbone.blocks.{}.", i)();
    const std::string a = p + "attention.";
    auto proj = [&](const std::string& nm, int N, int K) {
      return op.load_proj(w, a + nm + ".weight", a + nm + ".bias", N, K,
                          kKey);
    };
    b.q = proj("query", D, D);
    b.k = proj("key", KV, D);
    b.v = proj("value", KV, D);
    b.ck = proj("condition_key", KV, D);
    b.cv = proj("condition_value", KV, D);
    b.o = proj("output", D, D);
    b.gu = op.load_proj(w, p + "feed_forward.gate_up_proj.weight", "",
                        2 * F, D, kKey);
    b.dn = op.load_proj(w, p + "feed_forward.down.weight", "", D, F, kKey);
    b.an = t(p + "attention_norm.weight");
    b.kn = t(a + "key_norm.weight"); b.vn = t(a + "value_norm.weight");
    b.fn = t(p + "feed_forward_norm.weight");
    bool ok = true;
    for (const relayvsr::Proj* pr :
         {&b.q, &b.k, &b.v, &b.ck, &b.cv, &b.o, &b.gu, &b.dn}) {
      ok = ok && pr->valid();
    }
    for (const SharedBuffer* s : {&b.an, &b.kn, &b.vn, &b.fn}) {
      ok = ok && !s->empty();
    }
    if (!ok) {
      return fail(fmt("refvsr-flash: block {} is incomplete", i)());
    }
  }
  return m;
}

void
MetalRelayVsrDecoder::build_rope_(int h, int w)
{
  Scratch& s = *_sc;
  // The reference's frequencies: angles = float(position) * inv_freq,
  // inv_freq = 1 / 10000^(2i / d) in f32, cos / sin in the model dtype.
  auto tab = [&](int d, int pos, std::vector<float>* out) {
    for (int i = 0; i < d / 2; ++i) {
      const float e = (float)(2 * i) / (float)d;
      const float inv = 1.0f / std::pow(10000.0f, e);
      const float a = (float)pos * inv;
      out->push_back(relayvsr::round_elt(std::cos(a), false));
      out->push_back(relayvsr::round_elt(std::sin(a), false));
    }
  };
  std::vector<float> t0, tm1, th, tw;
  tab(16, 0, &t0);
  tab(16, -1, &tm1);
  for (int y = 0; y < h; ++y) { tab(24, y, &th); }
  for (int x = 0; x < w; ++x) { tab(24, x, &tw); }
  s.tab_t0 = relayvsr::upload_f32(_mc, t0);
  s.tab_tm1 = relayvsr::upload_f32(_mc, tm1);
  s.tab_h = relayvsr::upload_f32(_mc, th);
  s.tab_w = relayvsr::upload_f32(_mc, tw);
}

bool
MetalRelayVsrDecoder::ensure_scratch_(int ph, int pw, std::string* err)
{
  if (_sc && _sc->ph == ph && _sc->pw == pw) { return true; }
  _sc.reset();
  auto sc = std::make_unique<Scratch>();
  sc->ph = ph;
  sc->pw = pw;
  sc->h = ph / _cfg.latent_stride();
  sc->w = pw / _cfg.latent_stride();
  const std::size_t n = (std::size_t)sc->h * sc->w;
  const std::size_t n2 = (std::size_t)(ph / 2) * (pw / 2);
  const std::size_t n4 = (std::size_t)(ph / 4) * (pw / 4);
  const std::size_t kv = (std::size_t)(_cfg.kv_heads * _cfg.head_dim);
  const std::size_t D = (std::size_t)_cfg.dim;
  sc->n = (int)n;
  auto mk = [&](std::size_t elts) {
    return _mc->make_shared_buffer(std::max<std::size_t>(elts * 2, 16));
  };
  sc->e0 = mk(n2 * (std::size_t)_cfg.enc[0]);
  sc->e1 = mk(n4 * (std::size_t)_cfg.enc[1]);
  sc->e2 = mk(n * (std::size_t)_cfg.enc[2]);
  sc->x = mk(n * D);
  sc->y = mk(n * D);
  sc->q = mk(n * D);
  sc->a = mk(n * D);
  sc->o = mk(n * D);
  sc->gu = mk(n * 2 * (std::size_t)_cfg.ffn);
  sc->sw = mk(n * (std::size_t)_cfg.ffn);
  sc->hd = mk(n * (std::size_t)_cfg.head_width);
  sc->res = mk(n * 768);
  sc->kr_cur = mk(n * kv);
  sc->kr_prev = mk(n * kv);
  sc->cr = mk(n * (std::size_t)_cfg.cond_channels);
  for (int i = 0; i < 2; ++i) {
    sc->hk[i] = mk((std::size_t)_cfg.depth * n * kv);
    sc->hv[i] = mk((std::size_t)_cfg.depth * n * kv);
  }
  bool ok = true;
  for (const SharedBuffer* b :
       {&sc->e0, &sc->e1, &sc->e2, &sc->x, &sc->y, &sc->q, &sc->a, &sc->o,
        &sc->gu, &sc->sw, &sc->hd, &sc->res, &sc->kr_cur, &sc->kr_prev,
        &sc->cr, &sc->hk[0], &sc->hk[1], &sc->hv[0], &sc->hv[1]}) {
    ok = ok && !b->empty();
  }
  if (!ok) {
    if (err != nullptr) {
      *err = fmt("refvsr-flash: cannot allocate a {}x{} frame's working "
                 "set ({} MB)", pw, ph, scratch_bytes(_cfg, ph, pw) >> 20)();
    }
    return false;
  }
  _sc = std::move(sc);
  build_rope_(_sc->h, _sc->w);
  _in_window = 0;
  return true;
}

// ---- the ANE tier -----------------------------------------------------------

bool
MetalRelayVsrDecoder::ane_setup_(int rows)
{
  if (_ane_tried) { return _ane != nullptr; }
  _ane_tried = true;
  if (_mc->session() == nullptr || rows <= 0) { return false; }
  // Nothing to stage where an int8 tier holds every feed-forward: no
  // module is built, so none is held.
  bool any = false;
  for (const Block& b : _blocks) {
    any = any || (!b.gu.held() && !b.dn.held());
  }
  if (!any) {
    _mc->session()->info(
        fmt("relayvsr-decoder: the int8 tier holds every block's feed-forward, "
            "which the ANE cannot stage; it stays on the GPU"));
    return false;
  }
  AneFeedForward::Options o;
  o.session = _mc->session();
  o.mc      = _mc;
  o.tag     = "relayvsr-decoder";
  o.hidden  = _cfg.dim;
  o.ffn     = _cfg.ffn;
  o.seq     = rows;
  o.rows    = _cfg.ane_rows;
  o.chunk   = relayvsr::ane_chunk(rows);
  o.f16     = true;
  o.profile = std::getenv("VPIPE_RELAYVSR_ANE_PROFILE") != nullptr;
  _ane = AneFeedForward::create(o);
  return _ane != nullptr;
}

bool
MetalRelayVsrDecoder::ane_eligible_(int l) const
{
  if (_ane == nullptr) { return false; }
  const Block& b = _blocks[(std::size_t)l];
  return !b.gu.held() && !b.dn.held();
}

void
MetalRelayVsrDecoder::ane_stage_(int l)
{
  const Block& b = _blocks[(std::size_t)l];
  // gate_up is [gate; up], gate first.
  AneFfnSource gate, up, down;
  gate.w = &b.gu.w;
  up.w = &b.gu.w;
  up.offset = (std::size_t)_cfg.ffn;
  down.w = &b.dn.w;
  _ane->stage(l, gate, up, down, 0);
}

bool
MetalRelayVsrDecoder::prepare(const SharedBuffer& latent, int h4, int w4,
                              Condition* out, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (!ensure_scratch_(h4 * _cfg.latent_stride(), w4 * _cfg.latent_stride(),
                       err)) {
    return false;
  }
  Scratch& s = *_sc;
  _ops->tune_pending();
  const relayvsr::Ops& op = *_ops;
  const int n = s.n, D = _cfg.dim, KV = _cfg.kv_heads * _cfg.head_dim;
  const std::size_t layer_bytes = (std::size_t)n * KV * 2;
  if (out->k.empty() || out->k.byte_size() < layer_bytes * _cfg.depth) {
    out->k = _mc->make_shared_buffer(layer_bytes * _cfg.depth);
    out->v = _mc->make_shared_buffer(layer_bytes * _cfg.depth);
    if (!out->valid()) { return fail("condition allocation failed"); }
  }
  out->h = s.h;
  out->w = s.w;
  if (!_have_stream) {
    _stream = _mc->make_command_stream();
    _have_stream = true;
  }
  ComputeEncoder enc = _stream.begin_compute();
  enc.set_function(op.cond_rows);
  enc.set_buffer(0, latent); enc.set_buffer(1, s.cr);
  enc.set_constant(2, _cfg.cond_channels); enc.set_constant(3, n);
  enc.dispatch({(unsigned)_cfg.cond_channels, (unsigned)n, 1}, {64, 1, 1});
  op.gemm(enc, s.cr, 0, _cpw, &_cpb, s.x, 0, n, D, _cfg.cond_channels);
  enc.set_function(op.rms_embed);
  enc.set_buffer(0, s.x); enc.set_buffer(1, _cnorm); enc.set_buffer(2, _ctype);
  enc.set_buffer(3, s.x);
  enc.set_constant(4, D); enc.set_constant(5, _cfg.eps);
  enc.dispatch({256, (unsigned)n, 1}, {256, 1, 1});
  for (int l = 0; l < _cfg.depth; ++l) {
    const Block& b = _blocks[(std::size_t)l];
    const std::size_t off = (std::size_t)l * layer_bytes;
    enc.set_function(op.rms);
    enc.set_buffer(0, s.x); enc.set_buffer(1, b.an); enc.set_buffer(2, s.y);
    enc.set_constant(3, D); enc.set_constant(4, _cfg.eps);
    enc.dispatch({256, (unsigned)n, 1}, {256, 1, 1});
    // The condition's own K (normed, then rotated at t = 0 -- the position
    // every use of it will rotate it to) and V: what the windows read. And
    // the queries of every block but the last, whose K and V are all the
    // decoder reads of it.
    const bool last = l + 1 == _cfg.depth;
    std::vector<relayvsr::Ops::Out> outs = {{&b.ck, &s.kr_cur, 0},
                                            {&b.cv, &out->v, off}};
    if (!last) { outs.push_back({&b.q, &s.q, 0}); }
    op.proj_shared(enc, s.y, 0, n, outs);
    enc.set_function(op.rms_heads64);
    enc.set_buffer(0, s.kr_cur); enc.set_buffer(1, b.kn);
    enc.set_constant(2, KV); enc.set_constant(3, _cfg.kv_heads);
    enc.set_constant(4, _cfg.eps);
    enc.dispatch({32, (unsigned)n, (unsigned)_cfg.kv_heads}, {32, 1, 1});
    auto rope = [&](const SharedBuffer& in, std::size_t in_off, int ld_in,
                    const SharedBuffer& o, std::size_t o_off, int ld_o,
                    int heads, const SharedBuffer& tt) {
      enc.set_function(op.rope3d);
      enc.set_buffer(0, in, in_off); enc.set_buffer(1, o, o_off);
      enc.set_buffer(2, tt); enc.set_buffer(3, s.tab_h);
      enc.set_buffer(4, s.tab_w);
      enc.set_constant(5, ld_in); enc.set_constant(6, ld_o);
      enc.set_constant(7, s.w); enc.set_constant(8, heads);
      enc.dispatch({32, (unsigned)n, (unsigned)heads}, {32, 1, 1});
    };
    rope(s.kr_cur, 0, KV, out->k, off, KV, _cfg.kv_heads, s.tab_t0);
    enc.set_function(op.rms_heads64);
    enc.set_buffer(0, out->v, off); enc.set_buffer(1, b.vn);
    enc.set_constant(2, KV); enc.set_constant(3, _cfg.kv_heads);
    enc.set_constant(4, _cfg.eps);
    enc.dispatch({32, (unsigned)n, (unsigned)_cfg.kv_heads}, {32, 1, 1});
    if (last) { break; }
    rope(s.q, 0, D, s.q, 0, D, _cfg.heads, s.tab_t0);
    relayvsr::WindowPlane self;
    self.k = &out->k; self.k_off = off; self.k_tok = KV;
    self.v = &out->v; self.v_off = off; self.v_tok = KV;
    self.win_h = _cfg.win_self[0]; self.win_w = _cfg.win_self[1];
    op.window_attention(enc, s.q, 0, D, {self}, s.a, 0, D, s.h, s.w,
                        _cfg.heads, _cfg.kv_heads, _cfg.head_dim,
                        _cfg.sage_mode());
    op.proj(enc, s.a, 0, b.o, s.o, 0, n);
    enc.set_function(op.resadd);
    enc.set_buffer(0, s.x); enc.set_buffer(1, s.o); enc.set_buffer(2, s.x);
    enc.set_constant(3, n * D);
    enc.dispatch({(unsigned)(n * D), 1, 1}, {256, 1, 1});
    enc.set_function(op.rms);
    enc.set_buffer(0, s.x); enc.set_buffer(1, b.fn); enc.set_buffer(2, s.y);
    enc.set_constant(3, D); enc.set_constant(4, _cfg.eps);
    enc.dispatch({256, (unsigned)n, 1}, {256, 1, 1});
    op.proj(enc, s.y, 0, b.gu, s.gu, 0, n);
    enc.set_function(op.swiglu);
    enc.set_buffer(0, s.gu); enc.set_buffer(1, s.sw);
    enc.set_constant(2, _cfg.ffn);
    enc.set_constant(3, (unsigned)((std::size_t)n * _cfg.ffn));
    enc.dispatch({(unsigned)((std::size_t)n * _cfg.ffn), 1, 1}, {256, 1, 1});
    op.proj(enc, s.sw, 0, b.dn, s.o, 0, n);
    enc.set_function(op.resadd);
    enc.set_buffer(0, s.x); enc.set_buffer(1, s.o); enc.set_buffer(2, s.x);
    enc.set_constant(3, n * D);
    enc.dispatch({(unsigned)(n * D), 1, 1}, {256, 1, 1});
  }
  enc.end();
  std::string ge;
  if (!_stream.commit().wait_ok(&ge)) {
    return fail("refvsr-flash: priming a keyframe failed: " + ge);
  }
  return true;
}

bool
MetalRelayVsrDecoder::decode(const SharedBuffer& lr, int ph, int pw,
                             const Condition& start, const Condition& end,
                             int out_h, int out_w, bool f32,
                             SharedBuffer* out, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  const int ps = _cfg.pad_stride(), sc = _cfg.upscale;
  if (ph % ps != 0 || pw % ps != 0 || out_h <= 0 || out_w <= 0 ||
      out_h > sc * ph || out_w > sc * pw) {
    return fail("refvsr-flash: a frame of the wrong geometry");
  }
  if (!ensure_scratch_(ph, pw, err)) { return false; }
  Scratch& s = *_sc;
  if (!start.valid() || !end.valid() || start.h != s.h || start.w != s.w ||
      end.h != s.h || end.w != s.w) {
    return fail("refvsr-flash: the keyframes are not this frame's size");
  }
  // The accurate tier's split-K tuning, with no encoder open.
  _ops->tune_pending();
  const relayvsr::Ops& op = *_ops;
  const long i8_0 = op.i8_gemms(), dense_0 = op.dense_gemms();
  const int n = s.n, D = _cfg.dim, KV = _cfg.kv_heads * _cfg.head_dim;
  const std::size_t layer_bytes = (std::size_t)n * KV * 2;
  const std::size_t out_bytes = (std::size_t)3 * out_h * out_w * (f32 ? 4 : 1);
  if (out->empty() || out->byte_size() < out_bytes) {
    *out = _mc->make_shared_buffer(out_bytes);
    if (out->empty()) { return fail("frame allocation failed"); }
  }
  if (!_have_stream) {
    _stream = _mc->make_command_stream();
    _have_stream = true;
  }
  // History: the previous frame of this window, when there is one and the
  // checkpoint keeps it.
  const bool hist = _in_window > 0 && _cfg.history > 1;
  const int cur = s.cur, prev = 1 - s.cur;
  relayvsr::Profile prof("frame");
  ComputeEncoder enc = _stream.begin_compute();
  op.conv3x3(enc, lr, _e0w, _e0b, s.e0, ph, pw, 3, _cfg.enc[0], 2, true);
  op.conv3x3(enc, s.e0, _e1w, _e1b, s.e1, ph / 2, pw / 2, _cfg.enc[0],
             _cfg.enc[1], 2, true);
  // The third conv at the latent stride / 4: 1 at 4x, 2 at 2x.
  op.conv3x3(enc, s.e1, _e2w, _e2b, s.e2, ph / 4, pw / 4, _cfg.enc[1],
             _cfg.enc[2], _cfg.latent_stride() / 4, true);
  op.gemm(enc, s.e2, 0, _e3w, &_e3b, s.x, 0, n, D, _cfg.enc[2]);
  enc.set_function(op.rms_embed);
  enc.set_buffer(0, s.x); enc.set_buffer(1, _enorm); enc.set_buffer(2, _vtype);
  enc.set_buffer(3, s.x);
  enc.set_constant(4, D); enc.set_constant(5, _cfg.eps);
  enc.dispatch({256, (unsigned)n, 1}, {256, 1, 1});
  prof.mark(enc, "encoder");
  auto rope = [&](const SharedBuffer& in, std::size_t in_off, int ld_in,
                  const SharedBuffer& o, std::size_t o_off, int ld_o,
                  int heads, const SharedBuffer& tt) {
    enc.set_function(op.rope3d);
    enc.set_buffer(0, in, in_off); enc.set_buffer(1, o, o_off);
    enc.set_buffer(2, tt); enc.set_buffer(3, s.tab_h);
    enc.set_buffer(4, s.tab_w);
    enc.set_constant(5, ld_in); enc.set_constant(6, ld_o);
    enc.set_constant(7, s.w); enc.set_constant(8, heads);
    enc.dispatch({32, (unsigned)n, (unsigned)heads}, {32, 1, 1});
  };
  auto heads_norm = [&](const SharedBuffer& x, std::size_t off,
                        const SharedBuffer& g) {
    enc.set_function(op.rms_heads64);
    enc.set_buffer(0, x, off); enc.set_buffer(1, g);
    enc.set_constant(2, KV); enc.set_constant(3, _cfg.kv_heads);
    enc.set_constant(4, _cfg.eps);
    enc.dispatch({32, (unsigned)n, (unsigned)_cfg.kv_heads}, {32, 1, 1});
  };
  using Clk = std::chrono::steady_clock;
  auto ms_since = [](Clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clk::now() - t0).count();
  };
  const int F = _cfg.ffn;
  const bool ane_on = _cfg.ane_ffn && ane_setup_(n);
  _ane_rows = 0;
  for (int l = 0; l < _cfg.depth; ++l) {
    const Block& b = _blocks[(std::size_t)l];
    const std::size_t off = (std::size_t)l * layer_bytes;
    const AneFeedForward::Plan plan =
        ane_on && ane_eligible_(l) ? _ane->plan_block()
                                   : AneFeedForward::Plan::kGpu;
    const bool ane_block = plan == AneFeedForward::Plan::kSplit;
    const bool ane_probe = plan == AneFeedForward::Plan::kProbe;
    // A measured block drains at its split point, and the frame's commands
    // are encoded as one: whatever is queued before this block (the frame
    // encoder, unsplit blocks) is committed and waited for first, or the
    // drain's time is theirs (see AneFeedForward::needs_barrier).
    if ((ane_block || ane_probe) && (l == 0 || _ane->needs_barrier())) {
      std::string ge;
      if (!metal_compute::CommandStream::sync_and_reopen(enc, &ge)) {
        return fail("refvsr-flash: a frame failed: " + ge);
      }
    }
    if (ane_block) { ane_stage_(l); }
    enc.set_function(op.rms);
    enc.set_buffer(0, s.x); enc.set_buffer(1, b.an); enc.set_buffer(2, s.y);
    enc.set_constant(3, D); enc.set_constant(4, _cfg.eps);
    enc.dispatch({256, (unsigned)n, 1}, {256, 1, 1});
    op.proj_shared(enc, s.y, 0, n,
                   {{&b.q, &s.q, 0},
                    {&b.k, &s.hk[cur], off},
                    {&b.v, &s.hv[cur], off}});
    heads_norm(s.hk[cur], off, b.kn);
    heads_norm(s.hv[cur], off, b.vn);
    prof.mark(enc, "qkv");
    rope(s.q, 0, D, s.q, 0, D, _cfg.heads, s.tab_t0);
    rope(s.hk[cur], off, KV, s.kr_cur, 0, KV, _cfg.kv_heads, s.tab_t0);
    if (hist) {
      rope(s.hk[prev], off, KV, s.kr_prev, 0, KV, _cfg.kv_heads, s.tab_tm1);
    }
    std::vector<relayvsr::WindowPlane> planes;
    auto plane = [&](const SharedBuffer* k, std::size_t koff,
                     const SharedBuffer* v, std::size_t voff,
                     const int* win) {
      relayvsr::WindowPlane pl;
      pl.k = k; pl.k_off = koff; pl.k_tok = KV;
      pl.v = v; pl.v_off = voff; pl.v_tok = KV;
      pl.win_h = win[0]; pl.win_w = win[1];
      planes.push_back(pl);
    };
    plane(&start.k, off, &start.v, off, _cfg.win_start);
    plane(&end.k, off, &end.v, off, _cfg.win_end);
    if (hist) { plane(&s.kr_prev, 0, &s.hv[prev], off, _cfg.win_video); }
    plane(&s.kr_cur, 0, &s.hv[cur], off, _cfg.win_video);
    prof.mark(enc, "rope");
    op.window_attention(enc, s.q, 0, D, planes, s.a, 0, D, s.h, s.w,
                        _cfg.heads, _cfg.kv_heads, _cfg.head_dim,
                        _cfg.sage_mode());
    prof.mark(enc, "attn");
    op.proj(enc, s.a, 0, b.o, s.o, 0, n);
    enc.set_function(op.resadd);
    enc.set_buffer(0, s.x); enc.set_buffer(1, s.o); enc.set_buffer(2, s.x);
    enc.set_constant(3, n * D);
    enc.dispatch({(unsigned)(n * D), 1, 1}, {256, 1, 1});
    prof.mark(enc, "o-proj");
    enc.set_function(op.rms);
    enc.set_buffer(0, s.x); enc.set_buffer(1, b.fn); enc.set_buffer(2, s.y);
    enc.set_constant(3, D); enc.set_constant(4, _cfg.eps);
    enc.dispatch({256, (unsigned)n, 1}, {256, 1, 1});
    // The feed-forward; split with the Neural Engine, the ANE takes the
    // LAST rows of y into the same rows of o (fp16 both ways), once what
    // was encoded is drained.
    int a_rows = 0;
    double drain_ms = 0.0;
    if (ane_block || ane_probe) {
      const auto t0 = Clk::now();
      enc.end();
      metal_compute::CommandStream::Fence fence = _stream.commit();
      const bool staged = ane_block && _ane->join_stage(l);
      std::string ge;
      if (!fence.wait_ok(&ge)) {
        if (_ane) { _ane->join(); }
        return fail("refvsr-flash: a frame failed: " + ge);
      }
      drain_ms = ms_since(t0);
      if (staged) {
        a_rows = _ane->begin(s.y, s.o, n);
      } else if (ane_block && _mc->session() != nullptr) {
        _mc->session()->warn(fmt("relayvsr-decoder: staging block {}'s "
                                 "feed-forward for the ANE failed; it keeps "
                                 "the GPU", l));
      }
      enc = _stream.begin_compute();
      prof.host("ff-drain");
    }
    // The GPU's rows [r0, r0 + m).
    auto gpu_ff = [&](int r0, int m) {
      if (m <= 0) { return; }
      op.proj(enc, s.y, (std::size_t)r0 * D * 2, b.gu, s.gu,
              (std::size_t)r0 * 2 * F * 2, m);
      enc.set_function(op.swiglu);
      enc.set_buffer(0, s.gu, (std::size_t)r0 * 2 * F * 2);
      enc.set_buffer(1, s.sw, (std::size_t)r0 * F * 2);
      enc.set_constant(2, F);
      enc.set_constant(3, (unsigned)((std::size_t)m * F));
      enc.dispatch({(unsigned)((std::size_t)m * F), 1, 1}, {256, 1, 1});
      op.proj(enc, s.sw, (std::size_t)r0 * F * 2, b.dn, s.o,
              (std::size_t)r0 * D * 2, m);
    };
    const auto tg0 = Clk::now();
    gpu_ff(0, n - a_rows);
    if (a_rows > 0 || ane_probe) {
      std::string ge;
      if (!metal_compute::CommandStream::sync_and_reopen(enc, &ge)) {
        if (_ane) { _ane->join(); }
        return fail("refvsr-flash: a frame failed: " + ge);
      }
      const double gpu_ms = ms_since(tg0);
      if (a_rows > 0) {
        const bool ok = _ane->finish(l, n, gpu_ms, drain_ms);
        _ane_rows += a_rows;
        const int lost = ok ? -1 : _ane->lost_row0();
        if (lost >= 0 && lost < n) { gpu_ff(lost, n - lost); }
      } else {
        _ane->note_probe(drain_ms, gpu_ms);
      }
      prof.host(a_rows > 0 ? "ff-split" : "ff-probe");
    }
    enc.set_function(op.resadd);
    enc.set_buffer(0, s.x); enc.set_buffer(1, s.o); enc.set_buffer(2, s.x);
    enc.set_constant(3, n * D);
    enc.dispatch({(unsigned)(n * D), 1, 1}, {256, 1, 1});
    prof.mark(enc, "ff");
  }
  op.gemm(enc, s.x, 0, _hw, &_hb, s.hd, 0, n, _cfg.head_width, D);
  enc.set_function(op.silu);
  enc.set_buffer(0, s.hd);
  enc.set_constant(1, (unsigned)((std::size_t)n * _cfg.head_width));
  enc.dispatch({(unsigned)((std::size_t)n * _cfg.head_width), 1, 1},
               {256, 1, 1});
  op.gemm(enc, s.hd, 0, _rw, &_rb, s.res, 0, n, 768, _cfg.head_width);
  enc.set_function(op.frame_out);
  enc.set_buffer(0, s.res); enc.set_buffer(1, lr); enc.set_buffer(2, *out);
  enc.set_constant(3, ph); enc.set_constant(4, pw);
  enc.set_constant(5, out_h); enc.set_constant(6, out_w);
  enc.set_constant(7, f32 ? 1 : 0);
  enc.set_constant(8, sc);
  enc.dispatch({(unsigned)out_w, (unsigned)out_h, 3}, {32, 8, 1});
  prof.mark(enc, "head");
  enc.end();
  std::string ge;
  if (!_stream.commit().wait_ok(&ge)) {
    return fail("refvsr-flash: a frame failed: " + ge);
  }
  _i8_gemms = op.i8_gemms() - i8_0;
  _dense_gemms = op.dense_gemms() - dense_0;
  prof.report(_mc, fmt("{} rows{}, {} int8 / {} dense GEMMs, {} ANE rows", n,
                       hist ? ", with history" : "", _i8_gemms, _dense_gemms,
                       _ane_rows)());
  s.cur = prev;
  ++_in_window;
  return true;
}

}  // namespace genai
}  // namespace vpipe
