// metal-qwen-hc.cc -- Qwen3.8-Flash-Next (model_type qwen4_exp) on
// MetalQwenModel: the Qwen3.5-MoE backbone (gated-DeltaNet + gated full
// attention, 512 routed experts) with three things the Qwen3.5 code has
// none of. Reference: transformers' models/qwen4_exp.
//
// HYPER-CONNECTIONS. No pre-norms and no residual adds: the residual is
// hc_count streams of hidden ([rows, S*H]), the embedding copied into each.
// Every sublayer reads a learned mix of the normed streams and writes its
// output into each stream at a learned injection weight
// (Qwen4ExpTextGatedResidual; kernels in qwen4_hc.metal); a final mixer
// (the same block without injection) collapses the streams for the head,
// and there is no final norm.
//
// THE N-GRAM PLE (one linear layer). Each token's 2- and 3-gram (over an
// EOS-delimited history) is hashed into 16 heads of ~20M-row prime tables
// -- a 51B-parameter embedding, 38 GB at 5 bits -- whose rows are READ BY
// ROW from the checkpoint's shards through the page cache (ple_embed_):
// the table never resides. Projected to one key per stream and a shared
// value, gated by each stream's activations, plus a dilated depthwise conv
// over time whose history rides in the layer's own conv state.
//
// THE QSA INDEXER (full layers): each query attends only the top 512
// blocks of 4 keys. Attention is dense while a query sees at most budget +
// ratio - 1 = 2051 tokens, which is what this file runs; past that the
// output departs from the reference (warned once).
//
// The checkpoint's non-expert projections come in several widths and
// groups; every one is re-encoded to w8 at load, at group 64 -- or 32
// where a pack uses group 32 (Config::hc_group) -- losslessly: an affine
// code fits a byte, and a group of 64 or 128 is groups of 32 or 64 with
// the same scale. So the backbone runs the uniform-w8 kernels; a module a
// pack left unquantized runs dense. The router is dense (oQ5e) or w8 (MLX's
// packs); the routed experts stay at their stored width and group (4, 5, 6
// or 8 bits, group 32 or 64: expert_fns_); the zero-centred norms keep 1
// + w in f32, as the reference forms it.

#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/shared/dflash-drafter.h"
#include "generative-models/shared/i8-gemm.h"

#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/model-loader.h"
#include "generative-models/moe-expert-cache.h"
#include "generative-models/safetensors-ranges.h"
#include "generative-models/weight-set.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

namespace {

// From this K a w8 projection of 1-8 rows takes the split-K GEMV: the
// mixers' 10240 (qmv 27-37 us there, split 4-8). At 6144 and below the
// qmv already runs at the memory's pace and split-K is slower.
constexpr int kHcSplitK = 8192;

float
bf16_f32_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::uint16_t
f32_bf16_(float f)
{
  std::uint32_t b;
  std::memcpy(&b, &f, 4);
  b += 0x7fffu + ((b >> 16) & 1u);
  return (std::uint16_t)(b >> 16);
}

std::uint16_t
f32_elt_(float f, bool bf16)
{
  if (bf16) { return f32_bf16_(f); }
  const _Float16 h = (_Float16)f;
  std::uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}

float
elt_f32_(std::uint16_t u, bool bf16)
{
  if (bf16) { return bf16_f32_(u); }
  _Float16 h;
  std::memcpy(&h, &u, 2);
  return (float)h;
}

// A stored 16- or 32-bit float at `p` (index i) as f32.
float
stored_f32_(const void* p, std::size_t i, const std::string& dtype)
{
  if (dtype == "F32") {
    float f;
    std::memcpy(&f, static_cast<const std::uint8_t*>(p) + i * 4, 4);
    return f;
  }
  std::uint16_t u;
  std::memcpy(&u, static_cast<const std::uint8_t*>(p) + i * 2, 2);
  if (dtype == "BF16") { return bf16_f32_(u); }
  _Float16 h;
  std::memcpy(&h, &u, 2);
  return (float)h;
}

// Run f(lo, hi) over [0, n) in up to 16 chunks on the system's worker
// pool. A decode step's 16 n-gram rows are 48 small reads from the drive;
// one after another (rows under 64 ran on the caller) they took 1.3-3 ms
// of every token before its GPU work began, the drive answering one at a
// time. In flight together they take one read's time or so. (It spawned
// eight threads a call above that: GCD's pool is already running.)
template <typename F>
void
parallel_rows_(std::size_t n, F&& f)
{
  if (n < 2) { f((std::size_t)0, n); return; }
  struct Ctx {
    F*          f;
    std::size_t n, chunks;
  } ctx{&f, n, std::min<std::size_t>(n, 16)};
  dispatch_apply_f(ctx.chunks,
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   &ctx, [](void* p, std::size_t i) {
                     const Ctx& c = *static_cast<const Ctx*>(p);
                     const std::size_t lo = i * c.n / c.chunks;
                     const std::size_t hi = (i + 1) * c.n / c.chunks;
                     if (lo < hi) { (*c.f)(lo, hi); }
                   });
}

// Code j of an MLX affine row of `bits`-wide codes: a little-endian bit
// stream, code j at bits [bits * j, bits * j + bits).
inline unsigned
code_at_(const std::uint8_t* row, int bits, int j, std::size_t row_bytes)
{
  const std::size_t bit = (std::size_t)bits * (std::size_t)j;
  const std::size_t byte = bit >> 3;
  unsigned v = row[byte];
  if (byte + 1 < row_bytes) { v |= (unsigned)row[byte + 1] << 8; }
  return (v >> (bit & 7)) & ((1u << bits) - 1u);
}

}  // namespace

// ---- Loading ----------------------------------------------------------

namespace {

// A QSA batch's rows: as many as its block scores and picks keep within
// kQsaScratch (each row [blocks + width + 1] int32 / f32), 256 to 2048.
// One batch a 2048-row piece where they fit: its score, select and
// attention went eight 256-row dispatches a layer (~11 ms on an M5 Ultra,
// a whole piece's 8.1).
constexpr std::size_t kQsaScratch = 64u << 20;
inline int qsa_batch_rows_(std::int64_t ctx_tokens, int ratio, int sel_w)
{
  const std::size_t per = 4 * ((std::size_t)std::max<std::int64_t>(
                                   ctx_tokens, 0) / std::max(ratio, 1) +
                               (std::size_t)sel_w + 1);
  return (int)std::clamp<std::size_t>(kQsaScratch / per, 256, 2048);
}

struct HcLoad {
  WeightSet& ws;
  const MetalLlamaWeights& src;
  metal_compute::MetalCompute* mc;
  bool bf16;
  std::string* err;
  int G = 64;     // the group affine projections are re-encoded at

  bool fail(std::string m) const {
    if (err != nullptr && err->empty()) { *err = std::move(m); }
    return false;
  }

  SharedBuffer raw(const std::string& name) const {
    return ws.read(name, mc, WeightSet::Residency::Copied);
  }

  // The compute dtype, from a stored 16/32-bit tensor.
  SharedBuffer elt(const std::string& name) const {
    const auto* ti = src.info(name);
    if (ti == nullptr) { return {}; }
    SharedBuffer r = raw(name);
    if (r.empty()) { return {}; }
    const std::size_t n = ti->dtype == "F32" ? ti->nbytes / 4 : ti->nbytes / 2;
    if ((ti->dtype == "BF16" && bf16) || (ti->dtype == "F16" && !bf16)) {
      return r;
    }
    SharedBuffer o = mc->make_shared_buffer(n * 2);
    auto* d = static_cast<std::uint16_t*>(o.contents());
    for (std::size_t i = 0; i < n; ++i) {
      d[i] = f32_elt_(stored_f32_(r.contents(), i, ti->dtype), bf16);
    }
    return o;
  }

  SharedBuffer f32(const std::string& name, float add = 0.0f) const {
    const auto* ti = src.info(name);
    if (ti == nullptr) { return {}; }
    SharedBuffer r = raw(name);
    if (r.empty()) { return {}; }
    const std::size_t n = ti->dtype == "F32" ? ti->nbytes / 4 : ti->nbytes / 2;
    SharedBuffer o = mc->make_shared_buffer(n * 4);
    auto* d = static_cast<float*>(o.contents());
    for (std::size_t i = 0; i < n; ++i) {
      d[i] = stored_f32_(r.contents(), i, ti->dtype) + add;
    }
    return o;
  }

  // A zero-centred norm as the reference applies it: 1 + w, in f32.
  SharedBuffer norm1(const std::string& name) const { return f32(name, 1.0f); }

  // The projection `base` with input width K, as the forward runs it. An
  // AFFINE one (weight / scales / biases) is RE-ENCODED to w8 at the
  // model's group G: codes one byte each, scales and biases a group of G
  // (a stored group that is a multiple of G repeats its pair). Lossless for
  // every width up to 8 and every group G divides. A module the pack left
  // UNQUANTIZED (a weight and no scales) comes back in the compute dtype
  // with `s` and `b` empty, which `mm` runs dense.
  bool w8(const std::string& base, int K, SharedBuffer& w, SharedBuffer& s,
          SharedBuffer& b, int* rows = nullptr) const {
    const auto* wi = src.info(base + ".weight");
    const auto* si = src.info(base + ".scales");
    const auto* bi = src.info(base + ".biases");
    if (wi != nullptr && si == nullptr && bi == nullptr &&
        wi->shape.size() == 2 && wi->shape[1] == K &&
        wi->dtype != "U32") {
      if (rows != nullptr) { *rows = (int)wi->shape[0]; }
      w = elt(base + ".weight");
      s = {};
      b = {};
      return !w.empty() || fail(base + ": did not read");
    }
    if (wi == nullptr || si == nullptr || bi == nullptr ||
        wi->shape.size() != 2 || si->shape.size() != 2 || K <= 0) {
      return fail(base + ": not an affine projection");
    }
    const std::int64_t N = wi->shape[0];
    const std::int64_t cols = wi->shape[1], Gn = si->shape[1];
    if ((cols * 32) % K != 0 || K % Gn != 0) {
      return fail(fmt("{}: [{}, {}] codes do not fit {} inputs", base, N,
                      cols, K)());
    }
    const int bits = (int)(cols * 32 / K), group = (int)(K / Gn);
    if (bits < 2 || bits > 8 || group % G != 0 || K % G != 0) {
      return fail(fmt("{}: {}-bit group {} cannot be re-encoded to w8 at "
                      "group {}", base, bits, group, G)());
    }
    if (rows != nullptr) { *rows = (int)N; }
    SharedBuffer rw = raw(base + ".weight");
    SharedBuffer rs = raw(base + ".scales"), rb = raw(base + ".biases");
    if (rw.empty() || rs.empty() || rb.empty()) {
      return fail(base + ": did not read");
    }
    const int rep = group / G;
    const std::size_t gn = (std::size_t)K / G;
    s = mc->make_shared_buffer((std::size_t)N * gn * 2);
    b = mc->make_shared_buffer((std::size_t)N * gn * 2);
    auto* sd = static_cast<std::uint16_t*>(s.contents());
    auto* bd = static_cast<std::uint16_t*>(b.contents());
    for (std::int64_t r = 0; r < N; ++r) {
      for (std::size_t g = 0; g < gn; ++g) {
        const std::size_t src_i = (std::size_t)r * (std::size_t)Gn + g / rep;
        sd[(std::size_t)r * gn + g] =
            f32_elt_(stored_f32_(rs.contents(), src_i, si->dtype), bf16);
        bd[(std::size_t)r * gn + g] =
            f32_elt_(stored_f32_(rb.contents(), src_i, bi->dtype), bf16);
      }
    }
    if (bits == 8) {
      w = std::move(rw);
      return true;
    }
    w = mc->make_shared_buffer((std::size_t)N * (std::size_t)K);
    const auto* in = static_cast<const std::uint8_t*>(rw.contents());
    auto* out = static_cast<std::uint8_t*>(w.contents());
    const std::size_t rb_in = (std::size_t)cols * 4;
    parallel_rows_((std::size_t)N, [&](std::size_t lo, std::size_t hi) {
      for (std::size_t r = lo; r < hi; ++r) {
        const std::uint8_t* row = in + r * rb_in;
        std::uint8_t* o = out + r * (std::size_t)K;
        for (int j = 0; j < K; ++j) {
          o[j] = (std::uint8_t)code_at_(row, bits, j, rb_in);
        }
      }
    });
    return true;
  }

  // A w8 [rows, K] projection at group G dequantized into the compute
  // dtype: scale * q + bias in f32, rounded once -- what the reference's
  // bf16 run holds for an affine weight.
  SharedBuffer dense_of(const SharedBuffer& w, const SharedBuffer& s,
                        const SharedBuffer& b, int rows, int K) const {
    SharedBuffer o = mc->make_shared_buffer((std::size_t)rows * K * 2);
    const auto* q = static_cast<const std::uint8_t*>(w.contents());
    const auto* sp = static_cast<const std::uint16_t*>(s.contents());
    const auto* bp = static_cast<const std::uint16_t*>(b.contents());
    auto* d = static_cast<std::uint16_t*>(o.contents());
    const std::size_t gn = (std::size_t)K / G;
    for (std::size_t r = 0; r < (std::size_t)rows; ++r) {
      for (std::size_t j = 0; j < (std::size_t)K; ++j) {
        const std::size_t g = r * gn + j / G;
        const float v = elt_f32_(sp[g], bf16) * (float)q[r * K + j] +
                        elt_f32_(bp[g], bf16);
        d[r * K + j] = f32_elt_(v, bf16);
      }
    }
    return o;
  }

  // Several projections sharing input K, row-concatenated: w8 at group G
  // where every one is affine, the compute dtype where every one is dense,
  // and dense -- the affine ones dequantized -- where a pack mixes the two.
  bool fused(const std::vector<std::string>& bases, int K, SharedBuffer& w,
             SharedBuffer& s, SharedBuffer& b) const {
    std::vector<SharedBuffer> ws_, ss_, bs_;
    std::vector<int> rows_;
    bool any_dense = false, any_affine = false;
    for (const auto& nm : bases) {
      SharedBuffer a, c, d;
      int r = 0;
      if (!w8(nm, K, a, c, d, &r)) { return false; }
      (c.empty() ? any_dense : any_affine) = true;
      ws_.push_back(std::move(a));
      ss_.push_back(std::move(c));
      bs_.push_back(std::move(d));
      rows_.push_back(r);
    }
    if (any_dense && any_affine) {
      for (std::size_t i = 0; i < ws_.size(); ++i) {
        if (ss_[i].empty()) { continue; }
        ws_[i] = dense_of(ws_[i], ss_[i], bs_[i], rows_[i], K);
        ss_[i] = {};
        bs_[i] = {};
      }
    }
    std::size_t nw = 0, ng = 0;
    for (std::size_t i = 0; i < ws_.size(); ++i) {
      nw += ws_[i].byte_size();
      ng += ss_[i].byte_size();
    }
    w = mc->make_shared_buffer(nw);
    if (ng > 0) {
      s = mc->make_shared_buffer(ng);
      b = mc->make_shared_buffer(ng);
    } else {
      s = {};
      b = {};
    }
    std::size_t ow = 0, og = 0;
    for (std::size_t i = 0; i < ws_.size(); ++i) {
      std::memcpy((std::uint8_t*)w.contents() + ow, ws_[i].contents(),
                  ws_[i].byte_size());
      ow += ws_[i].byte_size();
      if (ng == 0) { continue; }
      std::memcpy((std::uint8_t*)s.contents() + og, ss_[i].contents(),
                  ss_[i].byte_size());
      std::memcpy((std::uint8_t*)b.contents() + og, bs_[i].contents(),
                  bs_[i].byte_size());
      og += ss_[i].byte_size();
    }
    return true;
  }

  // Two [rows, K] projections interleaved by row (row 2g = a's g, 2g + 1 =
  // b's g), as the fused SwiGLU kernels read gate|up. `wrow` / `grow` are
  // a row's code / scale bytes; `stacks` experts one after the other.
  static void interleave(const SharedBuffer& a, const SharedBuffer& b,
                         SharedBuffer& out, std::size_t stacks,
                         std::size_t rows, std::size_t rowb) {
    const auto* pa = static_cast<const std::uint8_t*>(a.contents());
    const auto* pb = static_cast<const std::uint8_t*>(b.contents());
    auto* po = static_cast<std::uint8_t*>(out.contents());
    for (std::size_t e = 0; e < stacks; ++e) {
      const std::size_t ib = e * rows * rowb, ob = e * 2 * rows * rowb;
      for (std::size_t g = 0; g < rows; ++g) {
        std::memcpy(po + ob + (2 * g) * rowb, pa + ib + g * rowb, rowb);
        std::memcpy(po + ob + (2 * g + 1) * rowb, pb + ib + g * rowb, rowb);
      }
    }
  }

  // (A template: the block type is the model's private HcBlock, named at
  // the call site, which has the access.)
  template <typename Block>
  bool hc_block(const std::string& p, int S, int H, int R, bool inject,
                Block& o) const {
    o.norm = norm1(p + "hc_norm.weight");
    if (o.norm.empty()) { return fail(p + "hc_norm: missing"); }
    // down and inject read the same normed streams: one matrix.
    const bool ok = inject
        ? fused({p + "input_mix_weight_down", p + "block_inject_weight"},
                S * H, o.dw, o.ds, o.db)
        : w8(p + "input_mix_weight_down", S * H, o.dw, o.ds, o.db);
    if (!ok || !w8(p + "input_mix_weight_up", R, o.uw, o.us, o.ub)) {
      return false;
    }
    o.inject = inject;
    return true;
  }
};

}  // namespace

bool
MetalQwenModel::build_layer_hc_(int L, std::string* err)
{
  const Config& c = _cfg;
  return build_hc_layer_(
      c.weight_prefix + c.model_seg + "layers." + std::to_string(L) + ".",
      c.layer_is_full(L), _layers[(std::size_t)L], err);
}

bool
MetalQwenModel::build_hc_layer_(const std::string& p, bool full, Layer& ly,
                                std::string* err)
{
  const Config& c = _cfg;
  const int H = c.hidden, S = c.hc_count, R = c.hc_lowrank;
  HcLoad ld{*_ws, _ws->src(), _mc, c.use_bf16, err, c.hc_group};
  ly.is_full = full;
  if (!ld.hc_block(p + "attn_hyper_connection.", S, H, R, true, ly.hc_attn) ||
      !ld.hc_block(p + "mlp_hyper_connection.", S, H, R, true, ly.hc_mlp)) {
    return false;
  }
  if (ly.is_full) {
    if (!ld.fused({p + "self_attn.q_proj", p + "self_attn.k_proj",
                   p + "self_attn.v_proj"}, H, ly.qw, ly.qs, ly.qb) ||
        !ld.w8(p + "self_attn.o_proj", c.qd(), ly.ow, ly.os, ly.ob)) {
      return false;
    }
    ly.q_norm = ld.norm1(p + "self_attn.q_norm.weight");
    ly.k_norm = ld.norm1(p + "self_attn.k_norm.weight");
    if (ly.q_norm.empty() || ly.k_norm.empty()) {
      return ld.fail(p + "q/k norm: missing");
    }
    if (c.qsa_heads > 0) {
      if (!ld.w8(p + "self_attn.indexer.index_qk_proj", H, ly.ixw,
                    ly.ixs, ly.ixb)) {
        return false;
      }
      ly.ix_qn = ld.norm1(p + "self_attn.indexer.q_layernorm.weight");
      ly.ix_kn = ld.norm1(p + "self_attn.indexer.k_layernorm.weight");
    }
  } else {
    if (!ld.fused({p + "linear_attn.in_proj_qkv", p + "linear_attn.in_proj_z",
                   p + "linear_attn.in_proj_a", p + "linear_attn.in_proj_b"},
                  H, ly.iqw, ly.iqs, ly.iqb) ||
        !ld.w8(p + "linear_attn.out_proj", c.value_dim(), ly.gow, ly.gos,
                  ly.gob)) {
      return false;
    }
    ly.conv_w = ld.elt(p + "linear_attn.conv1d.weight");   // [Cd, K, 1]
    ly.A_log = ld.f32(p + "linear_attn.A_log");
    ly.dt_bias = ld.f32(p + "linear_attn.dt_bias");
    ly.gdn_norm = ld.elt(p + "linear_attn.norm.weight");   // plain weight
    if (ly.conv_w.empty() || ly.A_log.empty() || ly.dt_bias.empty() ||
        ly.gdn_norm.empty()) {
      return ld.fail(p + "linear_attn: a tensor is missing");
    }
  }
  // MoE. The router: dense where the pack keeps mlp.gate unquantized
  // (oQ5e), else w8 at the model's group (MLX's packs store it 8-bit); the
  // experts at their stored width and group; the shared expert and its
  // gate w8 at the model's group, or dense where the pack left them so
  // (the 4-bit pack's MTP head).
  const int E = c.n_experts, I = c.moe_inner, Sh = c.moe_shared_inner;
  const int G = c.hc_group;
  if (_ws->src().info(p + "mlp.gate.scales") != nullptr) {
    if (!ld.w8(p + "mlp.gate", H, ly.rgw, ly.rgs, ly.rgb)) { return false; }
    ly.r_bits = 8;
  } else {
    ly.rgw = ld.elt(p + "mlp.gate.weight");
    ly.rgs = {};
    ly.rgb = {};
    ly.r_bits = 0;     // dense
    if (ly.rgw.empty()) { return ld.fail(p + "mlp.gate: missing"); }
  }
  const std::string sw = p + "mlp.switch_mlp.";
  const int gb = affine_bits_of_(sw + "gate_proj", H);
  const int ub = affine_bits_of_(sw + "up_proj", H);
  const int db = affine_bits_of_(sw + "down_proj", I);
  const int gg = affine_group_of_(sw + "gate_proj", H);
  const int ug = affine_group_of_(sw + "up_proj", H);
  const int dg = affine_group_of_(sw + "down_proj", I);
  if (gb != ub || gg != ug || gg != dg || !ensure_expert_fns_(gb, gg) ||
      !ensure_expert_fns_(db, dg)) {
    return ld.fail(fmt("{}: routed experts {}/{}/{}-bit, groups {}/{}/{} "
                       "(this build takes 4, 5, 6 or 8 bits at group 32 or "
                       "64, gate and up alike, one group)", sw, gb, ub, db,
                       gg, ug, dg)());
  }
  ly.eg_bits = ly.eu_bits = gb;
  ly.ed_bits = db;
  ly.e_group = gg;
  ly.moe_split = false;
  if (!_xstream) {
    // Resident: gate|up interleaved per expert, as the gathers read it.
    SharedBuffer gw = ld.raw(sw + "gate_proj.weight");
    SharedBuffer uw = ld.raw(sw + "up_proj.weight");
    SharedBuffer gs = ld.elt(sw + "gate_proj.scales");
    SharedBuffer us = ld.elt(sw + "up_proj.scales");
    SharedBuffer gbb = ld.elt(sw + "gate_proj.biases");
    SharedBuffer ubb = ld.elt(sw + "up_proj.biases");
    if (gw.empty() || uw.empty() || gs.empty() || us.empty() ||
        gbb.empty() || ubb.empty()) {
      return ld.fail(sw + ": the gate / up stacks did not read");
    }
    const std::size_t wrow = (std::size_t)H * gb / 8;
    const std::size_t grow = (std::size_t)(H / gg) * 2;
    ly.eguw = _mc->make_shared_buffer((std::size_t)E * 2 * I * wrow);
    ly.egus = _mc->make_shared_buffer((std::size_t)E * 2 * I * grow);
    ly.egub = _mc->make_shared_buffer((std::size_t)E * 2 * I * grow);
    HcLoad::interleave(gw, uw, ly.eguw, E, I, wrow);
    HcLoad::interleave(gs, us, ly.egus, E, I, grow);
    HcLoad::interleave(gbb, ubb, ly.egub, E, I, grow);
    ly.edw = ld.raw(sw + "down_proj.weight");
    ly.eds = ld.elt(sw + "down_proj.scales");
    ly.edb = ld.elt(sw + "down_proj.biases");
    if (ly.edw.empty() || ly.eds.empty() || ly.edb.empty()) {
      return ld.fail(sw + "down_proj: did not read");
    }
  }
  {
    // gate | up interleaved by row, as the SwiGLU kernels read it (dense
    // rows when the pack left them unquantized -- both, or the affine one
    // dequantized beside a dense other).
    SharedBuffer gw, gs, gbb, uw, us, ubb;
    if (!ld.w8(p + "mlp.shared_expert.gate_proj", H, gw, gs, gbb) ||
        !ld.w8(p + "mlp.shared_expert.up_proj", H, uw, us, ubb)) {
      return false;
    }
    if (gs.empty() != us.empty()) {
      if (!gs.empty()) { gw = ld.dense_of(gw, gs, gbb, Sh, H); }
      if (!us.empty()) { uw = ld.dense_of(uw, us, ubb, Sh, H); }
      gs = {};
      us = {};
      gbb = {};
      ubb = {};
    }
    const bool dense = gs.empty();
    const std::size_t wrow = (std::size_t)H * (dense ? 2 : 1);
    ly.sguw = _mc->make_shared_buffer(gw.byte_size() * 2);
    HcLoad::interleave(gw, uw, ly.sguw, 1, Sh, wrow);
    if (dense) {
      ly.sgus = {};
      ly.sgub = {};
    } else {
      const std::size_t grow = (std::size_t)(H / G) * 2;
      ly.sgus = _mc->make_shared_buffer(gs.byte_size() * 2);
      ly.sgub = _mc->make_shared_buffer(gbb.byte_size() * 2);
      HcLoad::interleave(gs, us, ly.sgus, 1, Sh, grow);
      HcLoad::interleave(gbb, ubb, ly.sgub, 1, Sh, grow);
    }
    ly.se_bits = dense ? 0 : 8;
  }
  if (!ld.w8(p + "mlp.shared_expert.down_proj", Sh, ly.sdw, ly.sds,
             ly.sdb) ||
      !ld.w8(p + "mlp.shared_expert_gate", H, ly.segw, ly.segs, ly.segb)) {
    return false;
  }
  ly.sd_bits = ly.sds.empty() ? 0 : 8;
  // The forward reads the shared expert and its gate one way or the other
  // (dense, or w8), the three together.
  if ((ly.se_bits == 0) != (ly.sd_bits == 0) ||
      (ly.se_bits == 0) != ly.segs.empty()) {
    return ld.fail(p + "mlp.shared_expert: its gate | up, down and gate are "
                       "not all quantized or all dense");
  }
  return true;
}

bool
MetalQwenModel::load_hc_mtp_(std::string* err)
{
  const Config& c = _cfg;
  const int H = c.hidden, S = c.hc_count, R = c.hc_lowrank;
  HcLoad ld{*_ws, _ws->src(), _mc, c.use_bf16, err, c.hc_group};
  auto M = std::make_unique<HcMtp>();
  if (!ld.w8("mtp.fc_embedding", H, M->few, M->fes, M->feb) ||
      !ld.w8("mtp.fc_hidden", H, M->fhw, M->fhs, M->fhb)) {
    return false;
  }
  M->ne = ld.norm1("mtp.pre_fc_norm_embedding.weight");
  M->nh = ld.norm1("mtp.pre_fc_norm_hidden.weight");
  if (M->ne.empty() || M->nh.empty()) {
    return ld.fail("mtp: a pre-fc norm is missing");
  }
  if (!ld.hc_block("mtp.hyper_connection_mixer.", S, H, R, false, M->mix)) {
    return false;
  }
  // The layer: built before the expert cache exists, which then takes its
  // experts as its last layer (or, with every expert resident, here).
  if (!build_hc_layer_("mtp.layers.0.", true, M->ly, err)) { return false; }
  // Its KV: one full layer, and its QSA keys on a pseudo layer as the
  // model's (ContextManager spec as load()'s, without linear layers).
  ContextManager::Spec cs;
  cs.metal = _mc;
  cs.n_layers = c.qsa_heads > 0 ? 2 : 1;
  cs.n_kv_heads = c.n_kv_heads;
  cs.head_dim = c.head_dim;
  cs.max_seq = c.max_seq;
  cs.page_tokens = c.page_tokens;
  cs.max_pages = c.max_pages;
  cs.is_linear_layer.assign((std::size_t)cs.n_layers, false);
  M->ctx = std::make_unique<ContextManager>(cs, nullptr);
  _hc_mtp = std::move(M);
  return true;
}

MetalQwenModel::HcMtp::Seq&
MetalQwenModel::hc_mtp_seq_(ContextId cid, int main_len)
{
  HcMtp& M = *_hc_mtp;
  HcMtp::Seq& q = M.seqs[cid.v];
  if (!q.cid.valid()) {
    q.cid = M.ctx->acquire_root();
    q.main_len = -1;
  }
  if (q.main_len != main_len) {
    // Out of step with the main context (a fresh one, a turn decoded
    // without the head, a rollback): start the head over, empty.
    const int len = M.ctx->seq_len_of(q.cid);
    if (len > 0) { M.ctx->kv_rollback(q.cid, len); }
    q.main_len = -1;
  }
  if (q.pending.empty()) {
    q.pending = _mc->make_shared_buffer(
        (std::size_t)_cfg.hc_count * _cfg.hidden * 2);
  }
  return q;
}

bool
MetalQwenModel::hc_mtp_prime_(ContextId cid, const SharedBuffer& resid,
                              std::span<const std::int32_t> ids,
                              int main_len, std::vector<float>* logits)
{
  const int n = (int)ids.size();
  if (_hc_mtp == nullptr || n <= 0) { return true; }
  const std::size_t row = (std::size_t)_cfg.hc_count * _cfg.hidden * 2;
  HcMtp::Seq& q = hc_mtp_seq_(cid, main_len - n);
  const bool pend = q.main_len >= 0 && main_len - n > 0;
  // The pairs: (the pending residual, ids[0]) when one waits, then (row t,
  // ids[t + 1]); the last row waits in turn.
  const int m = n - 1 + (pend ? 1 : 0);
  if (m > 0) {
    SharedBuffer in = _mc->make_shared_buffer((std::size_t)m * row);
    if (in.empty()) { return false; }
    auto* d = static_cast<std::uint8_t*>(in.contents());
    if (pend) { std::memcpy(d, q.pending.contents(), row); }
    std::memcpy(d + (pend ? row : 0), resid.contents(),
                (std::size_t)(n - 1) * row);
    HcRun r;
    r.mtp_in = &in;
    r.no_head = logits == nullptr;
    if (forward_hc_(q.cid, ids.subspan(pend ? 0 : 1, (std::size_t)m),
                    nullptr, nullptr, nullptr, -1, logits, &r).empty()) {
      q.main_len = -1;
      return false;
    }
  }
  std::memcpy(q.pending.contents(),
              static_cast<const std::uint8_t*>(resid.contents()) +
                  (std::size_t)(n - 1) * row, row);
  q.main_len = main_len;
  return true;
}

std::size_t
MetalQwenModel::hc_run_bytes(const Config& c, std::int64_t ctx_tokens)
{
  if (!c.hc()) { return 0; }
  // forward_hc_'s allocations at a prefill piece (2048 rows), element by
  // element: two bytes each but the f32 / int32 ones, marked.
  const std::size_t n = 2048;
  const std::size_t H = c.hidden, S = c.hc_count, R = c.hc_lowrank;
  const std::size_t SH = S * H, D = c.head_dim, qd = c.qd(), kd = c.kd();
  const std::size_t Nfq = 2 * qd + 2 * kd;
  const std::size_t Cd = c.gdn_conv_dim, vald = c.value_dim();
  const std::size_t Hv = c.gdn_v_heads, Nf = Cd + vald + 2 * Hv;
  const std::size_t E = c.n_experts, K = c.top_k, I = c.moe_inner;
  const std::size_t np = n * K;
  const std::size_t maxtiles = np / 32 + std::min(E, np), npad = maxtiles * 32;
  const std::size_t qW = (std::size_t)(c.qsa_heads + 1) * c.qsa_head_dim;
  std::size_t e2 = 0, e4 = 0;
  e2 += n * (4 * SH + 2 * R + S + 3 * H);         // X, hcn, hcu, resid ...
  e2 += n * (Nfq + 4 * kd + 6 * qd) + 32 * D;     // attention
  e2 += n * (Nf + 3 * vald + 2 * Hv + Cd);        // gated DeltaNet
  if (c.ple_layer >= 0) {
    e2 += n * ((SH + H) + 2 * SH + c.ple_dim);    // the n-gram rows
  }
  e2 += n * (E + 3 * H + c.moe_shared_inner + 1) + np * (1 + I + H);
  e2 += npad * (I + H);                           // the grouped GEMMs
  e2 += n * (qW + (std::size_t)c.qsa_heads * c.qsa_head_dim);
  e2 += (std::size_t)c.vocab;                     // the last row's logits
  e4 += n * (S + 2 * Hv + 3) + np + 3 * E + 2 * npad + maxtiles;
  const std::size_t ratio = std::max(1, c.qsa_ratio);
  const std::size_t sel_w = (std::size_t)c.qsa_budget + ratio - 1;
  e4 += (std::size_t)qsa_batch_rows_(ctx_tokens, (int)ratio, (int)sel_w) *
        ((std::size_t)std::max<std::int64_t>(ctx_tokens, 0) / ratio +
         sel_w + 1);                              // a sparse batch's picks
  const std::size_t fwd = 2 * e2 + 4 * e4;
  // K/V a token: the full layers, the QSA key layers (four full layers a
  // pseudo layer), the MTP head's layer and its key layer.
  int full = 0;
  for (int L = 0; L < c.n_layers; ++L) { full += c.layer_is_full(L); }
  const int qsa = c.qsa_heads > 0 ? (full + 3) / 4 : 0;
  const std::size_t kv_tok =
      (std::size_t)(full + qsa + 1 + (c.qsa_heads > 0 ? 1 : 0)) *
      c.n_kv_heads * D * 2 * 2;
  // A context's recurrent state, every linear layer, in three slots: the
  // SSM in f32, the conv tail (enlarged to carry the PLE's history).
  const std::size_t linear = (std::size_t)(c.n_layers - full);
  std::size_t conv = (std::size_t)(c.gdn_conv_kernel - 1) * Cd * 2;
  if (c.ple_layer >= 0) {
    conv += (std::size_t)(c.ple_conv_kernel - 1) * c.ple_ngram * SH * 2 + 8;
  }
  const std::size_t state =
      linear * (Hv * c.gdn_v_dim * c.gdn_k_dim * 4 + conv) * 3;
  return fwd + kv_tok * (std::size_t)std::max<std::int64_t>(ctx_tokens, 0) +
         state;
}

void
MetalQwenModel::forget_context(ContextId cid)
{
  if (_hc_mtp == nullptr) { return; }
  auto it = _hc_mtp->seqs.find(cid.v);
  if (it == _hc_mtp->seqs.end()) { return; }
  if (it->second.cid.valid()) { _hc_mtp->ctx->release(it->second.cid); }
  _hc_mtp->seqs.erase(it);
}

int
MetalQwenModel::affine_bits_of_(const std::string& base, int K) const
{
  const auto* wi = _ws->src().info(base + ".weight");
  if (wi == nullptr || wi->shape.empty() || K <= 0) { return 0; }
  return (int)((wi->shape.back() * 32) / K);
}

int
MetalQwenModel::affine_group_of_(const std::string& base, int K) const
{
  const auto* si = _ws->src().info(base + ".scales");
  if (si == nullptr || si->shape.empty() || si->shape.back() <= 0) {
    return 0;
  }
  return (int)(K / si->shape.back());
}

bool
MetalQwenModel::load_hc_table_(const std::string& base, SharedBuffer& w,
                               SharedBuffer& s, SharedBuffer& b,
                               std::string* err)
{
  const Config& c = _cfg;
  HcLoad ld{*_ws, _ws->src(), _mc, c.use_bf16, err, c.hc_group};
  return ld.w8(base, c.hidden, w, s, b);
}

std::string
MetalQwenModel::hc_layout() const
{
  if (!_cfg.hc() || _layers.empty()) { return {}; }
  const int G = _cfg.hc_group;
  auto moe = [&](const Layer& l) {
    const bool fz = l.eg_bits == l.ed_bits &&
                    fused_moe_fns_(l.eg_bits, l.e_group, G) != nullptr &&
                    l.se_bits == 8 && l.sd_bits == 8;
    return fmt("router {}, experts w{}g{}, shared expert {}{}",
               l.r_bits == 0 ? std::string("dense")
                             : fmt("w8g{}", G)(),
               l.eg_bits, l.e_group,
               l.se_bits == 0 ? std::string("dense")
                              : fmt("w8g{}", G)(),
               fz ? " (fused decode)" : "")();
  };
  const Layer& l0 = _layers[0];
  std::string s = fmt("projections w8g{}{}; {}", G,
                      l0.hc_attn.ds.empty() ? " (hyper-connections dense)"
                                            : "",
                      moe(l0))();
  if (_hc_mtp != nullptr) {
    const HcMtp& M = *_hc_mtp;
    s += fmt("; MTP head: fc {}, attention {}, {}",
             M.fhs.empty() ? "dense" : "w8",
             M.ly.qs.empty() ? "dense" : "w8", moe(M.ly))();
  }
  return s;
}

const MetalQwenModel::ExpertFns*
MetalQwenModel::expert_fns_(int bits, int group) const
{
  const auto it = _xfns.find(bits * 1000 + group);
  return it == _xfns.end() ? nullptr : &it->second;
}

bool
MetalQwenModel::ensure_expert_fns_(int bits, int group)
{
  const int key = bits * 1000 + group;
  if (_xfns.count(key) != 0) { return true; }
  using metal_compute::ComputeFunction;
  ExpertFns f;
  if (group == 64 && (bits == 4 || bits == 5 || bits == 8)) {
    // The handles every MoE model has dispatched by width.
    const bool w8 = bits == 8, w5 = bits == 5;
    f.gather_swiglu = w8 ? &_fn_moe_gather_swiglu_w8
                    : w5 ? &_fn_moe_gather_swiglu_w5 : &_fn_moe_gather_swiglu;
    f.gather_down = w8 ? &_fn_moe_gather_down_w8
                  : w5 ? &_fn_moe_gather_down_w5 : &_fn_moe_gather_down;
    f.grouped_swiglu = w8 ? &_fn_moe_grouped_swiglu_w8
                     : w5 ? &_fn_moe_grouped_swiglu_w5
                          : &_fn_moe_grouped_swiglu;
    f.grouped_down = w8 ? &_fn_moe_grouped_down_w8
                   : w5 ? &_fn_moe_grouped_down_w5 : &_fn_moe_grouped_down;
    f.qmm_swiglu = w8 ? &_fn_moe_qmm_grouped_swiglu_w8
                 : w5 ? &_fn_moe_qmm_grouped_swiglu_w5
                      : &_fn_moe_qmm_grouped_swiglu;
    f.qmm_down = w8 ? &_fn_moe_qmm_grouped_w8
               : w5 ? &_fn_moe_qmm_grouped_w5 : &_fn_moe_qmm_grouped;
    f.nax_swiglu = w8 ? &_fn_moe_nax_swiglu_w8
                 : w5 ? &_fn_moe_nax_swiglu_w5 : &_fn_moe_nax_swiglu_w4;
    f.nax_down = w8 ? &_fn_moe_nax_down_w8
               : w5 ? &_fn_moe_nax_down_w5 : &_fn_moe_nax_down_w4;
  } else if ((bits != 4 && bits != 5 && bits != 6 && bits != 8) ||
             (group != 32 && group != 64)) {
    return false;
  }
  const std::string g =
      "w" + std::to_string(bits) + "g" + std::to_string(group);
  auto own = [&](const metal_compute::ComputeLibrary& lib,
                 const std::string& name) -> const ComputeFunction* {
    if (!lib.valid()) { return nullptr; }
    _xfn_own.push_back(lib.function(name));
    return _xfn_own.back().valid() ? &_xfn_own.back() : nullptr;
  };
  if (f.gather_swiglu == nullptr) {
    f.gather_swiglu = own(_lib_qmv, "affine_gather_qmv_swiglu_" + g);
    f.gather_down = own(_lib_qmv, "affine_gather_down_qmv_" + g);
    f.grouped_swiglu = own(_lib_qmv, "affine_grouped_swiglu_" + g);
    f.grouped_down = own(_lib_qmv, "affine_grouped_down_" + g);
    f.qmm_swiglu = own(_lib_qmm, "affine_qmm_grouped_swiglu_" + g);
    f.qmm_down = own(_lib_qmm, "affine_qmm_grouped_" + g);
    // NAX steps one group of 64 a K step: a group of 32 takes steel.
    if (group == 64) {
      f.nax_swiglu =
          own(_lib_moe_nax, "affine_qmm_grouped_swiglu_nax_" + g);
      f.nax_down = own(_lib_moe_nax, "affine_qmm_grouped_nax_" + g);
    }
  }
  if (f.gather_swiglu == nullptr || f.gather_down == nullptr ||
      f.grouped_swiglu == nullptr || f.grouped_down == nullptr ||
      f.qmm_swiglu == nullptr || f.qmm_down == nullptr) {
    return false;
  }
  _xfns.emplace(key, f);
  // The hyper-connection decode's fused launch at this format, with the
  // shared expert at the model's group (where an instantiation exists;
  // w5g64 is the handle the oQ5e pack has always taken).
  if (_cfg.hc()) {
    const int sg = _cfg.hc_group;
    FusedMoeFns z;
    if (bits == 5 && group == 64 && sg == 64) {
      z.up = _fn_moe_up_fused.valid() ? &_fn_moe_up_fused : nullptr;
      z.down = _fn_moe_down_fused.valid() ? &_fn_moe_down_fused : nullptr;
    } else {
      const std::string sfx =
          g + (sg == 64 ? "" : "_sg" + std::to_string(sg));
      z.up = own(_lib_qmv, "affine_moe_up_fused_" + sfx);
      z.down = own(_lib_qmv, "affine_moe_down_fused_" + sfx);
    }
    if (z.up != nullptr && z.down != nullptr) {
      _xfused.emplace((bits * 1000 + group) * 1000 + sg, z);
    }
  }
  return true;
}

const MetalQwenModel::FusedMoeFns*
MetalQwenModel::fused_moe_fns_(int bits, int group, int sgroup) const
{
  const auto it = _xfused.find((bits * 1000 + group) * 1000 + sgroup);
  return it == _xfused.end() ? nullptr : &it->second;
}

bool
MetalQwenModel::load_hc_globals_(std::string* err)
{
  const Config& c = _cfg;
  const int H = c.hidden, S = c.hc_count, R = c.hc_lowrank;
  HcLoad ld{*_ws, _ws->src(), _mc, c.use_bf16, err, c.hc_group};
  const std::string root = c.weight_prefix + c.model_seg;
  if (!ld.hc_block(root + "hyper_connection_mixer.", S, H, R, false,
                   _hc_mix)) {
    return false;
  }
  if (c.ple_layer < 0) { return true; }
  auto P = std::make_unique<Ple>();
  const std::string p =
      root + "layers." + std::to_string(c.ple_layer) + ".ple.";
  if (!ld.fused({p + "key_proj", p + "value_proj"}, c.ple_dim, P->kvw,
                P->kvs, P->kvb)) {
    return false;
  }
  P->nk = ld.norm1(p + "norm_key.weight");
  P->nq = ld.norm1(p + "norm_query.weight");
  P->nc = ld.norm1(p + "norm_conv.weight");
  P->conv = ld.elt(p + "conv1d.weight");                 // [S*H, K, 1]
  if (P->nk.empty() || P->nq.empty() || P->nc.empty() || P->conv.empty()) {
    return ld.fail(p + ": a norm or the conv is missing");
  }
  // The hash: the checkpoint's own multipliers and per-head tables.
  auto i64 = [&](const std::string& nm, std::vector<std::int64_t>& out) {
    const auto* ti = _ws->src().info(nm);
    if (ti == nullptr || ti->dtype != "I64") { return false; }
    out.resize(ti->nbytes / 8);
    return _ws->src().pread_into(nm, out.data(), ti->nbytes, false);
  };
  const std::string e = p + "ple_embedding.";
  const int heads = (c.ple_ngram - 1) * c.ple_heads_per_ngram;
  if (!i64(e + "layer_multipliers", P->mult) ||
      !i64(e + "ngram_heads_vocab_sizes", P->sizes) ||
      !i64(e + "ngram_heads_offsets", P->offsets) ||
      (int)P->mult.size() < c.ple_ngram || (int)P->sizes.size() != heads ||
      (int)P->offsets.size() != heads) {
    return ld.fail(e + ": the n-gram hash tables are missing or malformed");
  }
  {
    SharedBuffer ws_ = ld.f32(e + "ngram_embedding.weight_scale");
    P->scale = ws_.empty() ? 1.0f
                           : static_cast<const float*>(ws_.contents())[0];
  }
  P->shard = e + "ngram_embedding.shards.";
  const auto* w0 = _ws->src().info(P->shard + "0.weight");
  const auto* s0 = _ws->src().info(P->shard + "0.scales");
  P->head_dim = c.ple_dim / heads;
  if (w0 == nullptr || w0->shape.size() != 2 || P->head_dim <= 0) {
    return ld.fail(e + ": the n-gram table shards are missing");
  }
  P->shard_rows = w0->shape[0];
  if (s0 == nullptr) {
    // Left unquantized (a pack whose group does not divide the rows' 160
    // values -- MLX skips such a table): the row as stored.
    P->bits = 0;
    P->dense_dtype = w0->dtype;
    const std::size_t esz = w0->dtype == "F32" ? 4 : 2;
    P->row_w = (std::size_t)w0->shape[1] * esz;
    if (w0->shape[1] != P->head_dim ||
        (w0->dtype != "BF16" && w0->dtype != "F16" && w0->dtype != "F32")) {
      return ld.fail(e + ": an n-gram table layout this reader does not "
                         "take");
    }
  } else {
    P->bits = (int)(w0->shape[1] * 32 / P->head_dim);
    P->group = s0->shape.size() == 2 ? (int)(P->head_dim / s0->shape[1]) : 0;
    P->row_w = (std::size_t)w0->shape[1] * 4;
    P->row_s = s0->shape.size() == 2 ? (std::size_t)s0->shape[1] * 2 : 0;
    P->scales_bf16 = s0->dtype == "BF16";
    if (P->bits < 2 || P->bits > 8 || P->group <= 0 ||
        (s0->dtype != "BF16" && s0->dtype != "F16")) {
      return ld.fail(e + ": an n-gram table layout this reader does not "
                         "take");
    }
  }
  P->rows = SafetensorsRanges::open(_ws->dir(), _ws->src(), err, true);
  if (!P->rows) { return false; }
  const int Cd = c.gdn_conv_dim, Kc = c.gdn_conv_kernel;
  P->hist = (c.ple_conv_kernel - 1) * c.ple_ngram;
  P->hist_off = (std::size_t)(Kc - 1) * Cd * 2;
  P->ids_off = P->hist_off + (std::size_t)P->hist * S * H * 2;
  _ple = std::move(P);
  return true;
}

// ---- The n-gram PLE rows ----------------------------------------------

bool
MetalQwenModel::ple_embed_(const SharedBuffer* state_in,
                           const SharedBuffer* state_out,
                           std::span<const std::int32_t> ids,
                           SharedBuffer& out, std::string* err)
{
  const Config& c = _cfg;
  const Ple& P = *_ple;
  const int n = (int)ids.size();
  const int heads = (int)P.sizes.size(), G = c.ple_ngram;
  const std::int64_t eos = c.ple_eos;
  // The context's last two tokens (stored + 1; 0 = none, which is EOS --
  // a fresh context's history is two EOS, as the reference pads it).
  if (state_in == nullptr || state_out == nullptr ||
      state_in->byte_size() < P.ids_off + 8 ||
      state_out->byte_size() < P.ids_off + 8) {
    if (err != nullptr) { *err = "no PLE state for this context"; }
    return false;
  }
  const auto* st = reinterpret_cast<const std::int32_t*>(
      static_cast<const std::uint8_t*>(state_in->contents()) + P.ids_off);
  auto* st_out = reinterpret_cast<std::int32_t*>(
      static_cast<std::uint8_t*>(state_out->contents()) + P.ids_off);
  std::vector<std::int64_t> h((std::size_t)n + 2);
  h[0] = st[0] > 0 ? st[0] - 1 : eos;
  h[1] = st[1] > 0 ? st[1] - 1 : eos;
  for (int t = 0; t < n; ++t) { h[(std::size_t)t + 2] = ids[(std::size_t)t]; }
  // Row ids [n, heads]: the reference's _shift_right_ignore_eos -- the
  // token `s` back is EOS when an EOS lies strictly between it and this
  // one -- then each order's mix of the shifted tokens, hashed per head.
  std::vector<std::int64_t> rows((std::size_t)n * heads);
  for (int t = 0; t < n; ++t) {
    const int pos = t + 2;
    std::int64_t sh[8];
    for (int s = 0; s < G; ++s) {
      bool cut = false;
      for (int q = pos - s + 1; q <= pos - 1; ++q) {
        if (h[(std::size_t)q] == eos) { cut = true; }
      }
      sh[s] = cut ? eos : h[(std::size_t)(pos - s)];
    }
    for (int g = 2; g <= G; ++g) {
      std::int64_t mix = sh[0] * P.mult[0];
      for (int k = 1; k < g; ++k) { mix ^= sh[k] * P.mult[(std::size_t)k]; }
      for (int j = 0; j < c.ple_heads_per_ngram; ++j) {
        const int hh = (g - 2) * c.ple_heads_per_ngram + j;
        const std::int64_t m = mix % P.sizes[(std::size_t)hh];
        rows[(std::size_t)t * heads + hh] =
            (m < 0 ? m + P.sizes[(std::size_t)hh] : m) +
            P.offsets[(std::size_t)hh];
      }
    }
  }
  // The rows, dequantized: [n, heads * head_dim] in the compute dtype.
  const int D = P.head_dim;
  out = _mc->make_shared_buffer((std::size_t)n * heads * D * 2);
  if (out.empty()) { return false; }
  auto* o = static_cast<std::uint16_t*>(out.contents());
  std::atomic<bool> ok{true};
  const bool bf16 = c.use_bf16;
  parallel_rows_(rows.size(), [&](std::size_t lo, std::size_t hi) {
    std::vector<std::uint8_t> w(P.row_w), s(P.row_s), b(P.row_s);
    for (std::size_t i = lo; i < hi && ok.load(); ++i) {
      const std::int64_t r = rows[i];
      const std::string base =
          P.shard + std::to_string(r / P.shard_rows) + ".";
      const std::size_t lr = (std::size_t)(r % P.shard_rows);
      if (!P.rows->read(base + "weight", lr * P.row_w, P.row_w, w.data())) {
        ok = false;
        return;
      }
      if (P.bits == 0) {
        // Unquantized: the stored value, as the reference's table holds
        // it, by weight_scale.
        for (int j = 0; j < D; ++j) {
          const float v = stored_f32_(w.data(), (std::size_t)j,
                                      P.dense_dtype) * P.scale;
          o[i * (std::size_t)D + (std::size_t)j] = f32_elt_(v, bf16);
        }
        continue;
      }
      if (!P.rows->read(base + "scales", lr * P.row_s, P.row_s, s.data()) ||
          !P.rows->read(base + "biases", lr * P.row_s, P.row_s, b.data())) {
        ok = false;
        return;
      }
      const std::string dt = P.scales_bf16 ? "BF16" : "F16";
      for (int j = 0; j < D; ++j) {
        const float sc = stored_f32_(s.data(), (std::size_t)(j / P.group), dt);
        const float bi = stored_f32_(b.data(), (std::size_t)(j / P.group), dt);
        const unsigned q = code_at_(w.data(), P.bits, j, P.row_w);
        // The reference's table holds the dequantized value in bf16 and
        // multiplies the gathered row by weight_scale; an f16 run rounds
        // once (a bf16 table is coarser than the f16 compute).
        const float dq = sc * (float)q + bi;
        const float v = (bf16 ? bf16_f32_(f32_bf16_(dq)) : dq) * P.scale;
        o[i * (std::size_t)D + (std::size_t)j] = f32_elt_(v, bf16);
      }
    }
  });
  if (!ok) {
    if (err != nullptr) { *err = "an n-gram table row did not read"; }
    return false;
  }
  st_out[0] = (std::int32_t)h[(std::size_t)n] + 1;
  st_out[1] = (std::int32_t)h[(std::size_t)n + 1] + 1;
  return true;
}

// ---- The forward ------------------------------------------------------

std::vector<float>
MetalQwenModel::prefill_hc_spliced(
    ContextId cid, const std::vector<float>& emb,
    const std::vector<std::int32_t>& ids,
    const std::vector<std::int32_t>& position_ids, int n,
    std::vector<float>* all_logits)
{
  const Config& c = _cfg;
  const int H = c.hidden;
  if (!c.hc() || n <= 0 || (int)ids.size() != n ||
      emb.size() < (std::size_t)n * H ||
      (!position_ids.empty() && position_ids.size() < (std::size_t)3 * n)) {
    return {};
  }
  const bool mrope = !position_ids.empty();
  const bool bf16 = c.use_bf16;
  auto store = [&](float f) -> std::uint16_t { return f32_elt_(f, bf16); };
  // In pieces, as prefill(): a piece's rows, and its mROPE tables
  // [rows, rotary_dim] (cat([f, f]) layout, each pair's axis by the
  // interleaved sections) -- the head primed piece by piece the same way.
  static const bool no_seed = std::getenv("VPIPE_MTP_NO_SEED") != nullptr;
  const bool prime = _hc_mtp != nullptr && _mtp_seed_enabled && !no_seed &&
                     _dflash == nullptr;
  const int rd = c.rotary_dim, half = rd / 2;
  const auto* invf = static_cast<const float*>(_inv_freq.contents());
  const int piece = std::min(prefill_piece_(), 2048);
  std::vector<float> r;
  SharedBuffer resid;
  HcRun run;
  if (prime) { run.resid = &resid; }
  if (all_logits != nullptr) { all_logits->clear(); }
  const HcTaps taps = hc_taps_begin_(cid, n, piece);
  for (int i0 = 0; i0 < n; i0 += piece) {
    const int m = std::min(piece, n - i0);
    hc_taps_piece_(taps, i0, m, run);
    SharedBuffer x = _mc->make_shared_buffer((std::size_t)m * H * 2);
    if (x.empty()) { return {}; }
    auto* xp = static_cast<std::uint16_t*>(x.contents());
    for (std::size_t k = 0; k < (std::size_t)m * H; ++k) {
      xp[k] = store(emb[(std::size_t)i0 * H + k]);
    }
    SharedBuffer cosb, sinb;
    if (mrope) {
      cosb = _mc->make_shared_buffer((std::size_t)m * rd * 2);
      sinb = _mc->make_shared_buffer((std::size_t)m * rd * 2);
      if (cosb.empty() || sinb.empty()) { return {}; }
      auto* cp = static_cast<std::uint16_t*>(cosb.contents());
      auto* sp = static_cast<std::uint16_t*>(sinb.contents());
      for (int t = 0; t < m; ++t) {
        for (int d = 0; d < half; ++d) {
          const int axis = d < (int)_mrope_axis.size() ? _mrope_axis[d] : 0;
          const float pos =
              (float)position_ids[(std::size_t)axis * n + i0 + t];
          const float ang = pos * invf[d];
          const std::uint16_t cc = store(std::cos(ang));
          const std::uint16_t ss = store(std::sin(ang));
          cp[(std::size_t)t * rd + d] = cc;
          cp[(std::size_t)t * rd + half + d] = cc;
          sp[(std::size_t)t * rd + d] = ss;
          sp[(std::size_t)t * rd + half + d] = ss;
        }
      }
    }
    std::vector<std::int32_t> ppos;
    if (mrope) {
      ppos.resize((std::size_t)3 * m);
      for (int a = 0; a < 3; ++a) {
        for (int t = 0; t < m; ++t) {
          ppos[(std::size_t)a * m + t] =
              position_ids[(std::size_t)a * n + i0 + t];
        }
      }
      run.mrope_pos = ppos.data();
    }
    const std::span<const std::int32_t> part(ids.data() + i0, (std::size_t)m);
    std::vector<float> rows;
    r = forward_hc_(cid, part, &x, mrope ? &cosb : nullptr,
                    mrope ? &sinb : nullptr, -1,
                    all_logits != nullptr ? &rows : nullptr, &run);
    run.mrope_pos = nullptr;
    if (all_logits != nullptr) {
      all_logits->insert(all_logits->end(), rows.begin(), rows.end());
    }
    if (r.empty() || !hc_taps_ingest_(taps, i0, m, run)) { return {}; }
    if (prime && !hc_mtp_prime_(cid, resid, part, _ctx->seq_len_of(cid),
                                nullptr)) {
      if (const SessionContextIntf* s = _mc->session()) {
        s->warn(fmt("[qwen] the MTP head's prompt priming failed; it "
                    "drafts from what it holds"));
      }
    }
  }
  return r;
}

bool
MetalQwenModel::hc_refuse_embeddings_()
{
  if (!_hc_embed_warned) {
    _hc_embed_warned = true;
    if (const SessionContextIntf* s = _mc->session()) {
      s->warn(fmt("[qwen] Qwen3.8-Flash-Next takes token ids (its n-gram "
                  "embedding hashes them): a prefill from embeddings alone "
                  "is refused -- an image or video turn goes through "
                  "prefill_hc_spliced, which carries them"));
    }
  }
  return false;
}

std::vector<float>
MetalQwenModel::prefill_all_logits(ContextId cid,
                                   const std::vector<std::int32_t>& ids,
                                   std::vector<float>* mtp_logits)
{
  if (!_cfg.hc() || ids.empty()) { return {}; }
  std::vector<float> all;
  SharedBuffer resid;
  HcRun r;
  const bool prime = _hc_mtp != nullptr && mtp_logits != nullptr;
  if (prime) { r.resid = &resid; }
  if (forward_hc_(cid, std::span<const std::int32_t>(ids), nullptr, nullptr,
                  nullptr, -1, &all, &r).empty()) {
    return {};
  }
  if (prime && !hc_mtp_prime_(cid, resid, ids, _ctx->seq_len_of(cid),
                              mtp_logits)) {
    return {};
  }
  return all;
}

std::vector<float>
MetalQwenModel::forward_hc_(ContextId cid, std::span<const std::int32_t> ids,
                            const SharedBuffer* embeds,
                            const SharedBuffer* mrope_cos,
                            const SharedBuffer* mrope_sin, int rope_pos,
                            std::vector<float>* all_logits,
                            const HcRun* run_in)
{
  static const HcRun kPlain;
  const HcRun& run = run_in != nullptr ? *run_in : kPlain;
  const Config& c = _cfg;
  const int n = (int)ids.size();
  if (n <= 0) { return {}; }
  if (run.tap_layers != nullptr &&
      (run.taps == nullptr || !_fn_hc_stream_mean.valid() ||
       run.tap_row0 < 0 || run.tap_row0 >= n ||
       run.taps->byte_size() < run.tap_layers->size() *
                                   (std::size_t)(n - run.tap_row0) *
                                   c.hidden * 2)) {
    return {};
  }
  // The stack: the model's own, or the MTP head's one layer over its own
  // context, its streams fused from the main model's (no PLE there).
  HcVerify* const verify = run.verify;
  const bool mtp = run.mtp_in != nullptr;
  if (mtp && _hc_mtp == nullptr) { return {}; }
  ContextManager& cx = mtp ? *_hc_mtp->ctx : *_ctx;
  Layer* const lys = mtp ? &_hc_mtp->ly : _layers.data();
  const int nL = mtp ? 1 : c.n_layers;
  const HcBlock& mixer = mtp ? _hc_mtp->mix : _hc_mix;
  const bool ple_on = _ple != nullptr && !mtp;
  const int H = c.hidden, S = c.hc_count, R = c.hc_lowrank, SH = S * H;
  const int D = c.head_dim, Hq = c.n_heads, Hkv = c.n_kv_heads;
  const int qd = c.qd(), kd = c.kd(), qdo = 2 * qd, Nfqkv = qdo + 2 * kd;
  const int Cd = c.gdn_conv_dim, keyd = c.key_dim(), vald = c.value_dim();
  const int Hv = c.gdn_v_heads, Dv = c.gdn_v_dim, Dk = c.gdn_k_dim;
  const int Hk = c.gdn_k_heads, Kc = c.gdn_conv_kernel;
  const int Nf = Cd + vald + 2 * Hv;
  const float eps = c.rms_eps;
  const float scale = 1.0f / std::sqrt((float)D);
  const SessionContextIntf* sess = _mc->session();
  ensure_decode_scratch_();
  tune_prefill_set_(n);
  // A linear layer's state in and out: ring slots 0 and 1 for a verify
  // that keeps its intermediate state, else the read / write slots (the
  // canonical buffer for both when the ring is off -- in place).
  const bool ring = verify != nullptr && verify->ring;
  auto conv_in = [&](int L) {
    return ring ? cx.conv_slot(cid, L, 0) : cx.conv_read(cid, L);
  };
  auto conv_out = [&](int L) {
    return ring ? cx.conv_slot(cid, L, 1) : cx.conv_write(cid, L);
  };
  auto ssm_in = [&](int L) {
    return ring ? cx.ssm_slot(cid, L, 0) : cx.ssm_read(cid, L);
  };
  auto ssm_out = [&](int L) {
    return ring ? cx.ssm_slot(cid, L, 1) : cx.ssm_write(cid, L);
  };

  // The PLE rows (CPU, from the table on disk) before anything is encoded
  // -- and before the KV slots, so a failed read leaves the context as it
  // was.
  // VPIPE_QWEN_HC_PROFILE: a one-row step's time split -- the PLE rows
  // (host reads), the encode loop (which waits on the GPU at every expert
  // cache sync), and the last commit's wait.
  static const bool kProf = std::getenv("VPIPE_QWEN_HC_PROFILE") != nullptr;
  using Clock = std::chrono::steady_clock;
  const auto tp0 = Clock::now();
  SharedBuffer ple_emb;
  if (ple_on) {
    std::string err;
    if (!ple_embed_(conv_in(c.ple_layer), conv_out(c.ple_layer), ids,
                    ple_emb, &err)) {
      if (sess != nullptr) { sess->warn(fmt("[qwen] PLE: {}", err)); }
      return {};
    }
  }
  // KV slots for the n rows.
  struct Chunk { std::size_t page_off; int slot, src_off, cnt; };
  std::vector<Chunk> chunks;
  int q_offset = -1;
  for (int written = 0; written < n;) {
    const int cap = cx.next_append_capacity(cid);
    const int cnt = std::min(n - written, cap);
    ContextManager::AppendSlot s = cx.append(cid, cnt);
    if (!s.valid()) { return {}; }
    if (q_offset < 0) { q_offset = s.position; }
    chunks.push_back({(std::size_t)s.page_id.v * cx.page_stride_bytes(),
                      s.slot_offset, written, cnt});
    written += cnt;
  }
  const int rpos0 = rope_pos >= 0 ? rope_pos : q_offset;
  const int page_tokens = cx.page_tokens();
  const int n_pages =
      cx.fill_page_table(cid, static_cast<std::int32_t*>(_pgtab.contents()));
  // The indexer ropes its queries and its blocks' keys at the rows' RoPE
  // positions (qpos, below): an image's 3-axis ones when the caller passes
  // them with its mROPE tables, else text positions.
  if (c.qsa_heads > 0 && mrope_cos != nullptr && run.mrope_pos == nullptr &&
      !_hc_qsa_warned) {
    _hc_qsa_warned = true;
    if (sess != nullptr) {
      sess->warn(fmt("[qwen] Qwen3.8-Flash-Next: the QSA indexer places "
                     "image tokens' blocks at text positions; past {} "
                     "tokens an image's keys rank approximately",
                     c.qsa_budget + c.qsa_ratio - 1));
    }
  }

  SharedBuffer xg;
  if (embeds == nullptr) { xg = _muxer->fetch_text(ids); }
  const SharedBuffer& x0 = embeds != nullptr ? *embeds : xg;
  if (x0.empty()) { return {}; }

  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  auto f32b = [&](std::size_t e) { return _mc->make_shared_buffer(e * 4); };
  // The forward's own scratch: kept between forwards, a slot a call in
  // order (each forward waits for its GPU work before it returns, so the
  // next finds them free) -- fifty allocations took ~0.3 ms of the CPU
  // before a step's first dispatch. VPIPE_QWEN_HC_SCRATCH=0 allocates
  // them anew. What outlives the forward (a verify's replay rows, the
  // residual handed out, members) is allocated as before.
  static const bool kKeep = [] {
    const char* e = std::getenv("VPIPE_QWEN_HC_SCRATCH");
    return e == nullptr || std::atoi(e) != 0;
  }();
  std::size_t slot = 0;
  auto scratch = [&](std::size_t bytes) {
    if (!kKeep) { return _mc->make_shared_buffer(bytes); }
    if (slot >= _hc_scratch.size()) { _hc_scratch.emplace_back(); }
    SharedBuffer& b = _hc_scratch[slot++];
    if (b.empty() || b.byte_size() < bytes) {
      b = _mc->make_shared_buffer(bytes);
    }
    return b.subview(0, bytes);
  };
  auto sbuf = [&](std::size_t e) { return scratch(e * 2); };
  auto sf32b = [&](std::size_t e) { return scratch(e * 4); };
  auto si32n = [&](std::size_t e) { return scratch(e * 4); };
  SharedBuffer X = sbuf((std::size_t)n * SH), hcn = sbuf((std::size_t)n * SH);
  SharedBuffer hcd = sbuf((std::size_t)n * R), hcu = sbuf((std::size_t)n * SH);
  SharedBuffer hcdi = sbuf((std::size_t)n * (R + S));
  SharedBuffer hci = sbuf((std::size_t)n * S), inj = sf32b((std::size_t)n * S);
  // A decode's and a verify's mixer blocks in two launches each (qwen4_hc
  // hcf_*: the six kernels' arithmetic, a block's injection pending into
  // the next block's norm), a verify's rows a grid slice each: four rows a
  // threadgroup normed and gathered every row of its slice -- 4x the work
  // and 20 KB of threadgroup memory -- and a verify of three ran 20.7 ms
  // a round against 18.5 so (M5 Ultra). VPIPE_QWEN_HCF_OFF=1 keeps the
  // six (an A/B); _RPS / _HPS (1, 2, 4) the down rows and up h a
  // simdgroup takes.
  static const bool kHcfOff = std::getenv("VPIPE_QWEN_HCF_OFF") != nullptr;
  auto hcf_knob = [](const char* k, int d) {
    const char* e = std::getenv(k);
    const int v = e != nullptr ? std::atoi(e) : d;
    return v == 1 ? 0 : v == 4 ? 2 : v == 2 ? 1 : (d == 1 ? 0 : 1);
  };
  static const int kHcfA = hcf_knob("VPIPE_QWEN_HCF_RPS", 1);
  static const int kHcfH = hcf_knob("VPIPE_QWEN_HCF_HPS", 1);
  // The fused kernels read w8 codes: every block this forward runs must be
  // quantized (a pack can leave them dense), and the choice holds for the
  // whole forward, since a block's injection is pending into the next.
  auto blk_q = [](const HcBlock& b) {
    return !b.ds.empty() && !b.us.empty();
  };
  bool blocks_q = true;
  if (mtp) {
    const HcMtp& Mh = *_hc_mtp;
    blocks_q = blk_q(Mh.ly.hc_attn) && blk_q(Mh.ly.hc_mlp) && blk_q(Mh.mix);
  } else {
    blocks_q = blk_q(_hc_mix);
    for (const Layer& lq : _layers) {
      blocks_q = blocks_q && blk_q(lq.hc_attn) && blk_q(lq.hc_mlp);
    }
  }
  const bool hcf =
      !kHcfOff && n <= 8 && S == 4 && H == 2560 &&   // kHcfBR's 5 * 512
      R % 64 == 0 && R <= 512 && H % (8 << kHcfH) == 0 && blocks_q &&
      _fn_hcf_pre[kHcfA].valid() && _fn_hcf_up[kHcfH].valid();
  SharedBuffer hcf_parts, inj2;
  if (hcf) {
    hcf_parts = sf32b((std::size_t)n * 8 * (R + S));
    inj2 = sf32b((std::size_t)n * S);
  }
  SharedBuffer hn = sbuf((std::size_t)n * H), y = sbuf((std::size_t)n * H);
  SharedBuffer qfull = sbuf((std::size_t)n * Nfqkv);
  SharedBuffer kbuf = sbuf((std::size_t)n * kd);
  SharedBuffer vbuf = sbuf((std::size_t)n * kd);
  SharedBuffer q3 = sbuf((std::size_t)n * qd);
  SharedBuffer gate3 = sbuf((std::size_t)n * qd);
  SharedBuffer qt = sbuf((std::size_t)n * qd + (std::size_t)32 * D);
  SharedBuffer kt = sbuf((std::size_t)n * kd), vt = sbuf((std::size_t)n * kd);
  SharedBuffer at = sbuf((std::size_t)n * qd), att = sbuf((std::size_t)n * qd);
  SharedBuffer mixqkv = sbuf((std::size_t)n * Nf);
  SharedBuffer zbuf = sbuf((std::size_t)n * vald);
  SharedBuffer abuf = sbuf((std::size_t)n * Hv);
  SharedBuffer bbuf = sbuf((std::size_t)n * Hv);
  SharedBuffer convout = sbuf((std::size_t)n * Cd);
  SharedBuffer gbuf = sf32b((std::size_t)n * Hv);
  SharedBuffer betabuf = sf32b((std::size_t)n * Hv);
  SharedBuffer ygdn = sbuf((std::size_t)n * vald);
  SharedBuffer normout = sbuf((std::size_t)n * vald);
  SharedBuffer kvp, gv, gvn_own;
  if (ple_on) {
    kvp = sbuf((std::size_t)n * (SH + H));
    gv = sbuf((std::size_t)n * SH);
    gvn_own = sbuf((std::size_t)n * SH);
    if (verify != nullptr) { verify->gvn = buf((std::size_t)n * SH); }
  }
  const SharedBuffer& gvn =
      verify != nullptr && ple_on ? verify->gvn : gvn_own;
  // MoE scratch -- the shapes and tiers forward_chunk_ uses.
  const int K = c.top_k, np = n * K;
  auto i32n = [&](std::size_t e) {
    return _mc->make_shared_buffer(e * sizeof(std::int32_t));
  };
  SharedBuffer moe_logits = sbuf((std::size_t)n * c.n_experts);
  SharedBuffer moe_eid = si32n((std::size_t)np), moe_w = sbuf((std::size_t)np);
  SharedBuffer moe_act = sbuf((std::size_t)np * c.moe_inner);
  SharedBuffer moe_part = sbuf((std::size_t)np * H);
  SharedBuffer moe_out = sbuf((std::size_t)n * H);
  SharedBuffer moe_ssg = sbuf((std::size_t)n * c.moe_shared_inner);
  SharedBuffer moe_sout = sbuf((std::size_t)n * H);
  SharedBuffer moe_gate = sbuf((std::size_t)n);
  const bool grouped = [&] {
    if (const char* e = std::getenv("VPIPE_QWEN_MOE_GROUPED")) {
      return std::atoi(e) != 0;
    }
    return n >= 64;
  }();
  const bool steel = grouped && [&] {
    if (const char* e = std::getenv("VPIPE_QWEN_MOE_STEEL")) {
      return std::atoi(e) != 0;
    }
    return moe_steel_(n);
  }();
  const int maxm = steel ? 32 : 2;
  const int moe_x = _xc ? _xc->slots() : c.n_experts;
  const int maxtiles = grouped ? (np + maxm - 1) / maxm + std::min(moe_x, np)
                               : 0;
  const int npad = maxtiles * maxm;
  auto gi = [&](std::size_t e) { return grouped ? si32n(e) : SharedBuffer{}; };
  SharedBuffer g_hist = gi((std::size_t)moe_x), g_boff = gi((std::size_t)moe_x);
  SharedBuffer g_curs = gi((std::size_t)moe_x);
  SharedBuffer g_t2e = gi((std::size_t)maxtiles), g_ntile = gi(1);
  SharedBuffer g_srow = gi((std::size_t)npad), g_sdst = gi((std::size_t)npad);
  SharedBuffer g_gact = grouped ? sbuf((std::size_t)npad * c.moe_inner)
                                : SharedBuffer{};
  SharedBuffer g_gdout = (grouped && !steel) ? sbuf((std::size_t)npad * H)
                                             : SharedBuffer{};
  // QSA scratch: the index projection, its normed + RoPE'd queries, and
  // per sparse-row batch the block scores and the selected tokens.
  // VPIPE_QWEN_QSA_OFF: dense attention at every length (the A/B that
  // shows what the indexer does past budget + ratio - 1 tokens).
  static const bool kQsaOff = std::getenv("VPIPE_QWEN_QSA_OFF") != nullptr;
  const bool qsa = c.qsa_heads > 0 && !kQsaOff;
  const int qh = c.qsa_heads, qdim = c.qsa_head_dim, ratio = c.qsa_ratio;
  const int qW = (qh + 1) * qdim, kblk = qsa ? c.qsa_budget / ratio : 0;
  const int thr = qsa ? c.qsa_budget + ratio - 1
                      : std::numeric_limits<int>::max();
  const int r_s = qsa ? std::clamp(thr - q_offset, 0, n) : n;
  const int sel_w = kblk * ratio + ratio - 1;
  // The dense rows of a piece that also has sparse ones, compact.
  const bool part_dense = qsa && r_s > 0 && r_s < n;
  SharedBuffer qtd, atd;
  if (part_dense) {
    qtd = sbuf((std::size_t)r_s * qd + (std::size_t)32 * D);
    atd = sbuf((std::size_t)r_s * qd);
  }
  const int nb_cap = (q_offset + n) / std::max(ratio, 1);
  const int qrb =
      std::min(qsa_batch_rows_(q_offset + n, ratio, sel_w), n - r_s);
  SharedBuffer ixb, ixq, qs_sc, qs_sel, qs_nsel, qpos, qs_part;
  // A decode step's and a verify's sparse rows: their selected keys in
  // chunks over many threadgroups, then combined (qsa_attn_part_f16).
  // VPIPE_QWEN_QSA_PART=0 keeps a threadgroup a (head, row) (an A/B).
  static const bool kQsaPart = [] {
    const char* e = std::getenv("VPIPE_QWEN_QSA_PART");
    return e == nullptr || std::atoi(e) != 0;
  }();
  constexpr int kQsaChunk = 64;            // qwen4_hc.metal's
  const int qs_nch = (sel_w + kQsaChunk - 1) / kQsaChunk;
  const bool qpart = qsa && kQsaPart && n <= 8 && D <= 256 &&
                     qs_nch <= 64 && _fn_qsa_attn_part.valid() &&
                     _fn_qsa_attn_comb.valid();
  if (qsa) {
    ixb = sbuf((std::size_t)n * qW);
    // Padded by a score tile's rows: qsa_score_mma_f16 reads to its end.
    ixq = sbuf((std::size_t)(n + 32) * qh * qdim);
    // Each row's RoPE position, three axes: text rows the same on all.
    qpos = si32n((std::size_t)n * 3);
    auto* qp = static_cast<std::int32_t*>(qpos.contents());
    for (int i = 0; i < n; ++i) {
      for (int a = 0; a < 3; ++a) {
        qp[(std::size_t)i * 3 + a] =
            run.mrope_pos != nullptr
                ? run.mrope_pos[(std::size_t)a * n + i] : rpos0 + i;
      }
    }
    if (_hc_axis.empty()) {
      const int half = c.rotary_dim / 2;
      _hc_axis = i32n((std::size_t)half);
      auto* ax = static_cast<std::int32_t*>(_hc_axis.contents());
      for (int d = 0; d < half; ++d) {
        ax[d] = d < (int)_mrope_axis.size() ? _mrope_axis[(std::size_t)d] : 0;
      }
    }
    if (qrb > 0) {
      qs_sc = sf32b((std::size_t)qrb * nb_cap);
      qs_sel = si32n((std::size_t)qrb * sel_w);
      qs_nsel = si32n((std::size_t)qrb);
      if (qpart) {
        qs_part = sf32b((std::size_t)qrb * Hq * qs_nch * (2 + D));
      }
    }
  }
  std::vector<int> full_ord((std::size_t)nL, -1);
  for (int L = 0, f = 0; L < nL; ++L) {
    if (lys[L].is_full) { full_ord[(std::size_t)L] = f++; }
  }
  const int n_out = all_logits != nullptr || run.all_rows ? n : 1;
  // VPIPE_QWEN_HC_DUMP=<dir>: the residual after the layers the golden
  // keeps (0, 1, 2, 3, 7, 23, last), every row, raw f32 resid_L<l>.f32 --
  // the debugging view that finds the first layer a run departs at.
  // VPIPE_QWEN_QSA_DUMP=<file>: the last row's QSA selection per full
  // layer alone, [12][sel_w tokens + count] int32 (what the tests read).
  const char* const kDump = std::getenv("VPIPE_QWEN_HC_DUMP");
  const char* const kSelDump = std::getenv("VPIPE_QWEN_QSA_DUMP");
  std::vector<int> dump_layers;
  std::vector<SharedBuffer> dumps;
  SharedBuffer dump_sel;      // the last row's QSA tokens per full layer
  if (kDump != nullptr && !mtp) {
    dump_layers = {0, 1, 2, 3, 7, 23, c.n_layers - 1};
    for (std::size_t i = 0; i < dump_layers.size(); ++i) {
      dumps.push_back(buf((std::size_t)n * SH));
    }
  }
  if ((kDump != nullptr || kSelDump != nullptr) && !mtp && qsa &&
      r_s < n) {
    dump_sel = i32n((std::size_t)16 * (sel_w + 1));
    std::memset(dump_sel.contents(), 0xff, dump_sel.byte_size());
  }
  if (_hc_logits.empty() ||
      _hc_logits.byte_size() < (std::size_t)n_out * c.vocab * 2) {
    _hc_logits = buf((std::size_t)n_out * c.vocab);
  }

  const auto tp1 = Clock::now();
  // One stream for every pass (a fresh queue's first commit costs 1-2 ms).
  metal_compute::CommandStream& stream = spec_stream_();
  {
    ComputeEncoder enc = stream.begin_compute();
    // y[rows, N] = xin[rows, Kk] @ W^T: a w8 projection at the model's
    // group (qmv a row, the matrix-core dense GEMM over a dequantized copy
    // on an M5 past its floor, else the steel GEMM) or a dense one (no
    // scales; a verify's few rows each as a decode step's one, as the
    // dense router's).
    auto mm = [&](const SharedBuffer& w, const SharedBuffer& s,
                  const SharedBuffer& b, const SharedBuffer& xin,
                  const SharedBuffer& yo, int Kk, int N, int rows) {
      if (s.empty()) {
        if (rows == 1) {
          dense_gemv_(enc, w, xin, 0, yo, 0, Kk, N);
        } else if (rows <= 8) {
          enc.set_function(_fn_hc_dense_gemv_rows);
          enc.set_buffer(0, xin); enc.set_buffer(1, w);
          enc.set_buffer(2, yo);
          enc.set_constant(3, Kk); enc.set_constant(4, N);
          enc.dispatch({32u * (unsigned)rows,
                        (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
        } else {
          dense_gemm_(enc, w, xin, yo, Kk, N, rows);
        }
        return;
      }
      // A LONG row (the mixers' 4 x 2560): K split across a threadgroup's
      // simdgroups, its runs added in order -- 3.5-8x the qmv a decode row,
      // 8-12x the batched GEMV at a verify's four, and a row the same bits
      // alone or batched, so the verify still keeps the decode's tokens.
      // ...and a prefill's injection (4 outputs: a matrix tile is 128 or
      // 256 wide), the same kernel a few rows a threadgroup.
      if (Kk >= kHcSplitK && (rows <= 8 || N <= 16)) {
        const bool few = N <= 64;
        enc.set_function(few ? _fn_hc_qmv8_sk16 : _fn_hc_qmv8_sk8);
        enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
        enc.set_buffer(3, xin); enc.set_buffer(4, yo);
        enc.set_constant(5, Kk); enc.set_constant(6, N);
        enc.set_constant(7, rows);
        const unsigned ks = few ? 16 : 8;
        enc.dispatch({32, ks * (unsigned)N, (unsigned)((rows + 3) / 4)},
                     {32, ks, 1});
        return;
      }
      // A verify's two to four rows through the decode's own qmv, a row a
      // threadgroup column (grid.x = 32 rows): its sums the decode's by
      // construction, the rows' weight reads meeting in the cache. The
      // batched GEMV's four rows a simdgroup ran the 6144 -> 2560 output
      // projection in 41 us at two rows (qmv 22) and 52 at four (31) on an
      // M5 Ultra. Past 128 MB of weights (the 636 MB LM head) the rows'
      // reads miss the cache -- 2.2 ms at four against the batch's 0.61
      // -- and the batch stays. VPIPE_QWEN_QMV_ROWS=0 keeps it everywhere
      // (an A/B).
      static const bool kQmvRows = [] {
        const char* e = std::getenv("VPIPE_QWEN_QMV_ROWS");
        return e == nullptr || std::atoi(e) != 0;
      }();
      if (rows == 1 || (kQmvRows && rows <= 4 &&
                        (std::size_t)N * (std::size_t)Kk <= (128u << 20))) {
        enc.set_function(_fn_qmv);
        enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
        enc.set_buffer(3, xin); enc.set_buffer(4, yo);
        enc.set_constant(5, Kk); enc.set_constant(6, N);
        enc.dispatch({32u * (unsigned)rows, (unsigned)((N + 3) / 4), 1},
                     {32, 2, 1});
        return;
      }
      if (rows <= 8) {
        // A few rows (a verify): the batched GEMV, each row's sums as the
        // one-row qmv's -- the verify keeps the decode's tokens.
        enc.set_function(_fn_hc_qmv8_b4);
        enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
        enc.set_buffer(3, xin); enc.set_buffer(4, yo);
        enc.set_constant(5, Kk); enc.set_constant(6, N);
        enc.set_constant(7, rows);
        enc.dispatch({32, (unsigned)((N + 3) / 4), (unsigned)((rows + 3) / 4)},
                     {32, 2, 1});
        return;
      }
      if (_use_mma && rows >= _mma_min_m) {
        // Dynamic-int8 (opt-in, LOSSY): the checkpoint's w8 codes straight
        // into the int8 GEMM, no dequant pass and no requant.
        // (Its kernels read scales a group of 64: at 32 the weight takes
        // the dequant below.)
        if (_i8 && c.hc_group == 64 &&
            _i8->gemm_affine(enc, xin, 0, w, s, b, 8, yo, 0, rows, N, Kk)) {
          ++_i8_count;
          return;
        }
        const std::size_t need = (std::size_t)N * Kk * 2;
        if (_w_deq.empty() || _w_deq.byte_size() < need) {
          _w_deq = _mc->make_shared_buffer(need);
        }
        enc.set_function(_fn_dequant8);
        enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
        enc.set_buffer(3, _w_deq);
        enc.set_constant(4, Kk); enc.set_constant(5, N);
        enc.dispatch({(unsigned)(Kk / 4), (unsigned)N, 1}, {64, 1, 1});
        dense_gemm_(enc, _w_deq, xin, yo, Kk, N, rows);
        return;
      }
      enc.set_function(_fn_qmm);
      enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
      enc.set_buffer(3, xin); enc.set_buffer(4, yo);
      enc.set_constant(5, Kk); enc.set_constant(6, N);
      enc.set_constant(7, rows);
      enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                    (unsigned)(((rows + 31) / 32) * 2), 2}, {32, 2, 2});
    };
    // Grouped RMS with f32 (1 + w): `groups` H-wide groups a row.
    auto grms = [&](const SharedBuffer& xin, const SharedBuffer& w,
                    const SharedBuffer& yo, int Hd, int groups, int rows) {
      // A prefill's rows: 8 values a load (a decode step and a verify
      // keep the other kernel's sums).
      const bool v8 = n > 8 && Hd % 8 == 0 && Hd <= 4096 &&
                      _fn_hc_rms8.valid();
      enc.set_function(v8 ? _fn_hc_rms8 : _fn_hc_rms);
      enc.set_buffer(0, xin); enc.set_buffer(1, w); enc.set_buffer(2, yo);
      enc.set_constant(3, Hd); enc.set_constant(4, groups);
      enc.set_constant(5, eps);
      if (v8) {
        enc.dispatch({64, (unsigned)groups, (unsigned)rows}, {64, 1, 1});
      } else {
        enc.dispatch({256, (unsigned)groups, (unsigned)rows}, {256, 1, 1});
      }
    };
    auto rms = [&](const SharedBuffer& xin, std::size_t xoff,
                   const SharedBuffer& w, const SharedBuffer& yo,
                   std::size_t yoff, int Rr, int Hd) {
      enc.set_function(rms_fn_(Hd, xoff, yoff));
      enc.set_buffer(0, xin, xoff); enc.set_buffer(1, w);
      enc.set_buffer(2, yo, yoff);
      enc.set_constant(3, Hd); enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)Rr, 1}, {256, 1, 1});
    };
    auto hslice = [&](const SharedBuffer& in, const SharedBuffer& out,
                      int Hh, int Ss, int W, int off, int block = 0,
                      int gstride = 0) {
      enc.set_function(_fn_head_slice);
      enc.set_buffer(0, in); enc.set_buffer(1, out);
      enc.set_constant(2, Hh); enc.set_constant(3, Ss);
      enc.set_constant(4, W); enc.set_constant(5, off);
      enc.set_constant(6, block); enc.set_constant(7, gstride);
      enc.dispatch({(unsigned)(Hh * W), 1, 1}, {256, 1, 1});
    };
    auto transpose = [&](const SharedBuffer& in, const SharedBuffer& out,
                         int A, int Bd) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, in); enc.set_buffer(1, out);
      enc.set_constant(2, A); enc.set_constant(3, Bd); enc.set_constant(4, D);
      enc.dispatch({(unsigned)D, (unsigned)Bd, (unsigned)A},
                   {(unsigned)D, 1, 1});
    };
    auto rope = [&](const SharedBuffer& xb, int heads) {
      if (mrope_cos != nullptr) {
        enc.set_function(_fn_mrope);
        enc.set_buffer(0, xb); enc.set_buffer(1, *mrope_cos);
        enc.set_buffer(2, *mrope_sin);
        enc.set_constant(3, heads); enc.set_constant(4, n);
        enc.set_constant(5, D); enc.set_constant(6, c.rotary_dim);
      } else {
        enc.set_function(_fn_rope_partial);
        enc.set_buffer(0, xb); enc.set_buffer(1, _inv_freq);
        enc.set_constant(2, heads); enc.set_constant(3, n);
        enc.set_constant(4, D); enc.set_constant(5, c.rotary_dim);
        enc.set_constant(6, rpos0);
      }
      enc.dispatch({(unsigned)(c.rotary_dim / 2), (unsigned)n, (unsigned)heads},
                   {(unsigned)(c.rotary_dim / 2), 1, 1});
    };
    auto kv_write = [&](const SharedBuffer& src, const SharedBuffer& pool) {
      for (const Chunk& ch : chunks) {
        enc.set_function(_fn_kv_write_paged);
        enc.set_buffer(0, src); enc.set_buffer(1, pool, ch.page_off);
        enc.set_constant(2, page_tokens); enc.set_constant(3, D);
        enc.set_constant(4, n); enc.set_constant(5, ch.src_off);
        enc.set_constant(6, ch.slot);
        enc.dispatch({(unsigned)D, (unsigned)ch.cnt, (unsigned)Hkv},
                     {(unsigned)D, 1, 1});
      }
    };
    // The fused blocks' pending injection: a sublayer's output and its
    // weights, added to X by the next block (or hc_flush, where anything
    // else reads X). A block writes its weights into the buffer the
    // pending one is not in.
    const SharedBuffer* pend_y = nullptr;
    const SharedBuffer* pend_inj = &inj;
    const SharedBuffer* blk_inj = &inj;
    bool inj_alt = true;
    auto hc_flush = [&]() {
      if (pend_y == nullptr) { return; }
      enc.set_function(_fn_hc_inject);
      enc.set_buffer(0, X); enc.set_buffer(1, *pend_y);
      enc.set_buffer(2, *pend_inj);
      enc.set_constant(3, H); enc.set_constant(4, S);
      enc.dispatch({(unsigned)(H / 8), (unsigned)S, (unsigned)n},
                   {64, 1, 1});
      pend_y = nullptr;
    };
    // Hyper-connection gated residual: the mixed input into `xmix` [n, H];
    // the injection weights into `inj` (when the block injects).
    auto hc_pre = [&](const HcBlock& b, const SharedBuffer& xmix) {
      if (hcf) {
        const SharedBuffer& wi = inj_alt ? inj2 : inj;
        const int N = b.inject ? R + S : R;
        const int pend = pend_y != nullptr ? 1 : 0;
        // Unread without a pending injection: any buffer binds.
        const SharedBuffer& py = pend != 0 ? *pend_y : X;
        const SharedBuffer& pi = pend != 0 ? *pend_inj : wi;
        const int zt = n;   // a row a slice
        const int rps = 1 << kHcfA, hps = 1 << kHcfH;
        const int G = (N + 4 * rps - 1) / (4 * rps);
        enc.set_function(_fn_hcf_pre[kHcfA]);
        enc.set_buffer(0, X); enc.set_buffer(1, py); enc.set_buffer(2, pi);
        enc.set_buffer(3, b.norm); enc.set_buffer(4, hcn);
        enc.set_buffer(5, b.dw); enc.set_buffer(6, b.ds);
        enc.set_buffer(7, b.db); enc.set_buffer(8, hcf_parts);
        enc.set_constant(9, H); enc.set_constant(10, N);
        enc.set_constant(11, n); enc.set_constant(12, eps);
        enc.set_constant(13, pend);
        enc.dispatch({(unsigned)(256 * G), (unsigned)S, (unsigned)zt},
                     {256, 1, 1});
        enc.set_function(_fn_hcf_up[kHcfH]);
        enc.set_buffer(0, hcf_parts); enc.set_buffer(1, b.uw);
        enc.set_buffer(2, b.us); enc.set_buffer(3, b.ub);
        enc.set_buffer(4, hcn); enc.set_buffer(5, xmix);
        enc.set_buffer(6, wi); enc.set_buffer(7, X);
        enc.set_buffer(8, py); enc.set_buffer(9, pi);
        enc.set_constant(10, H); enc.set_constant(11, R);
        enc.set_constant(12, N); enc.set_constant(13, n);
        enc.set_constant(14, pend);
        enc.dispatch({(unsigned)(256 * (H / (8 * hps))), 1u, (unsigned)zt},
                     {256, 1, 1});
        pend_y = nullptr;
        if (b.inject) {
          blk_inj = &wi;
          inj_alt = !inj_alt;
        }
        return;
      }
      grms(X, b.norm, hcn, H, S, n);
      if (b.inject && n > 8) {
        // A prefill: the matrix's down rows and its injection's as views,
        // apart -- one 324-wide GEMM pads a sixth 64-wide int8 tile, the
        // injection's GEMV is cheaper.
        // (A row: w8 codes and its group's scales, or the compute dtype.)
        const bool dq = b.ds.empty();
        const std::size_t wr = (std::size_t)SH * (dq ? 2 : 1);
        const std::size_t gr = dq ? 0 : (std::size_t)SH / c.hc_group * 2;
        auto sv = [&](const SharedBuffer& t, std::size_t o, std::size_t k) {
          return dq ? SharedBuffer{} : t.subview(o, k);
        };
        mm(b.dw.subview(0, R * wr), sv(b.ds, 0, R * gr),
           sv(b.db, 0, R * gr), hcn, hcd, SH, R, n);
        mm(b.dw.subview(R * wr, S * wr), sv(b.ds, R * gr, S * gr),
           sv(b.db, R * gr, S * gr), hcn, hci, SH, S, n);
        enc.set_function(_fn_hc_act);
        enc.set_buffer(0, hcd); enc.set_buffer(1, hci);
        enc.set_buffer(2, inj);
      } else if (b.inject) {
        // down | inject in one projection, then split: silu(d / S) into
        // hcd (the up projection's input), 2 sigmoid(ir / S) into inj.
        mm(b.dw, b.ds, b.db, hcn, hcdi, SH, R + S, n);
        enc.set_function(_fn_hc_act_fused);
        enc.set_buffer(0, hcdi); enc.set_buffer(1, hcd);
        enc.set_buffer(2, inj);
      } else {
        mm(b.dw, b.ds, b.db, hcn, hcd, SH, R, n);
        enc.set_function(_fn_hc_act);
        enc.set_buffer(0, hcd); enc.set_buffer(1, hcd);
        enc.set_buffer(2, inj);
      }
      enc.set_constant(3, R); enc.set_constant(4, S); enc.set_constant(5, n);
      enc.set_constant(6, b.inject && n > 8 ? 1 : 0);
      enc.dispatch({(unsigned)(n * (R + S)), 1, 1}, {256, 1, 1});
      mm(b.uw, b.us, b.ub, hcd, hcu, R, SH, n);
      enc.set_function(_fn_hc_mix);
      enc.set_buffer(0, hcu); enc.set_buffer(1, hcn); enc.set_buffer(2, xmix);
      enc.set_constant(3, H); enc.set_constant(4, S);
      enc.dispatch({(unsigned)(H / 8), (unsigned)n, 1}, {64, 1, 1});
    };
    auto hc_post = [&](const SharedBuffer& yo) {
      if (hcf) {
        pend_y = &yo;
        pend_inj = blk_inj;
        return;
      }
      enc.set_function(_fn_hc_inject);
      enc.set_buffer(0, X); enc.set_buffer(1, yo); enc.set_buffer(2, inj);
      enc.set_constant(3, H); enc.set_constant(4, S);
      enc.dispatch({(unsigned)(H / 8), (unsigned)S, (unsigned)n},
                   {64, 1, 1});
    };

    if (mtp) {
      // The head's streams: fc_hidden of each stream of the normed main
      // residual (one norm over all hc*H), plus fc_embedding of the normed
      // embedding of the next token, added to every stream (the injection
      // with weights 1).
      const HcMtp& M = *_hc_mtp;
      grms(*run.mtp_in, M.nh, hcn, SH, 1, n);
      mm(M.fhw, M.fhs, M.fhb, hcn, X, H, H, n * S);
      grms(x0, M.ne, hn, H, 1, n);
      mm(M.few, M.fes, M.feb, hn, y, H, H, n);
      std::fill_n(static_cast<float*>(inj.contents()), (std::size_t)n * S,
                  1.0f);
      hc_post(y);
    } else {
      // The embedding into every stream.
      enc.set_function(_fn_hc_tile);
      enc.set_buffer(0, x0); enc.set_buffer(1, X);
      enc.set_constant(2, H); enc.set_constant(3, S);
      enc.dispatch({(unsigned)H, (unsigned)S, (unsigned)n}, {256, 1, 1});
    }

    const bool use_pset = prefill_set_serves_(n);
    for (int L = 0; L < nL; ++L) {
      Layer& ly = lys[L];
      if (ple_on && L == c.ple_layer) {
        hc_flush();
        // X += gv + silu(conv(gvn)): the n-gram rows gated per stream.
        const SharedBuffer& cs = *conv_in(L);
        const SharedBuffer& cs_out = *conv_out(L);
        mm(_ple->kvw, _ple->kvs, _ple->kvb, ple_emb, kvp, c.ple_dim, SH + H,
           n);
        enc.set_function(_fn_ple_gate);
        enc.set_buffer(0, kvp); enc.set_buffer(1, X);
        enc.set_buffer(2, _ple->nk); enc.set_buffer(3, _ple->nq);
        enc.set_buffer(4, _ple->nc);
        enc.set_buffer(5, gv); enc.set_buffer(6, gvn);
        enc.set_constant(7, H); enc.set_constant(8, S);
        enc.set_constant(9, eps);
        enc.dispatch({256, (unsigned)S, (unsigned)n}, {256, 1, 1});
        enc.set_function(_fn_ple_conv);
        enc.set_buffer(0, X); enc.set_buffer(1, gv); enc.set_buffer(2, gvn);
        enc.set_buffer(3, cs, _ple->hist_off);
        enc.set_buffer(4, _ple->conv);
        enc.set_constant(5, SH); enc.set_constant(6, n);
        enc.set_constant(7, c.ple_conv_kernel);
        enc.set_constant(8, c.ple_ngram);
        enc.dispatch({(unsigned)SH, (unsigned)n, 1}, {256, 1, 1});
        enc.set_function(_fn_ple_state);
        enc.set_buffer(0, cs, _ple->hist_off); enc.set_buffer(1, gvn);
        enc.set_buffer(2, cs_out, _ple->hist_off);
        enc.set_constant(3, SH); enc.set_constant(4, n);
        enc.set_constant(5, _ple->hist);
        enc.dispatch({(unsigned)SH, 1, 1}, {256, 1, 1});
      }
      hc_pre(ly.hc_attn, hn);
      if (ly.is_full) {
        const SharedBuffer& kp = *cx.kpool(L);
        const SharedBuffer& vp = *cx.vpool(L);
        mm(ly.qw, ly.qs, ly.qb, hn, qfull, H, Nfqkv, n);
        // A decode step's and a verify's rows: the slices, norms,
        // transposes, ropes and K / V writes in one launch, each value as
        // those compute it (VPIPE_QWEN_ATTN_FUSED=0 keeps them).
        static const bool kAttnFused = [] {
          const char* e = std::getenv("VPIPE_QWEN_ATTN_FUSED");
          return e == nullptr || std::atoi(e) != 0;
        }();
        const bool af = kAttnFused && n <= 8 && D == 256 &&
                        mrope_cos == nullptr && c.rotary_dim <= D &&
                        _fn_hc_attn_prework.valid() &&
                        _fn_hc_attn_post.valid();
        if (af) {
          struct KvRows { std::uint32_t rb[8]; } kvr{};
          for (const Chunk& ch : chunks) {
            for (int t = ch.src_off; t < ch.src_off + ch.cnt; ++t) {
              kvr.rb[t] = (std::uint32_t)(ch.page_off / 2 +
                                          (std::size_t)(ch.slot + t -
                                                        ch.src_off) * D);
            }
          }
          enc.set_function(_fn_hc_attn_prework);
          enc.set_buffer(0, qfull); enc.set_buffer(1, ly.q_norm);
          enc.set_buffer(2, ly.k_norm); enc.set_buffer(3, _inv_freq);
          enc.set_buffer(4, qt); enc.set_buffer(5, kp);
          enc.set_buffer(6, vp);
          enc.set_constant(7, n); enc.set_constant(8, Nfqkv);
          enc.set_constant(9, Hq); enc.set_constant(10, Hkv);
          enc.set_constant(11, c.rotary_dim); enc.set_constant(12, rpos0);
          enc.set_constant(13, page_tokens); enc.set_constant(14, eps);
          enc.set_constant(15, kvr);
          enc.dispatch({256, (unsigned)(Hq + 2 * Hkv), (unsigned)n},
                       {256, 1, 1});
        } else {
          hslice(qfull, q3, n * Hq, 2 * D, D, 0, Hq, Nfqkv);
          hslice(qfull, gate3, n * Hq, 2 * D, D, D, Hq, Nfqkv);
          hslice(qfull, kbuf, n, Nfqkv, kd, qdo);
          hslice(qfull, vbuf, n, Nfqkv, kd, qdo + kd);
          grms(q3, ly.q_norm, q3, D, 1, n * Hq);
          grms(kbuf, ly.k_norm, kbuf, D, 1, n * Hkv);
          transpose(q3, qt, n, Hq);
          transpose(kbuf, kt, n, Hkv);
          transpose(vbuf, vt, n, Hkv);
          rope(qt, Hq);
          rope(kt, Hkv);
          kv_write(kt, kp);
          kv_write(vt, vp);
        }
        if (qsa) {
          // This layer's index keys: raw into the pseudo K pool, the blocks
          // this chunk completes pooled into the V pool.
          const int f = full_ord[(std::size_t)L];
          const SharedBuffer& xkp = *cx.kpool(nL + f / 4);
          const SharedBuffer& xvp = *cx.vpool(nL + f / 4);
          const int xhead = (f % 4) / 2, xoff = ((f % 4) % 2) * qdim;
          mm(ly.ixw, ly.ixs, ly.ixb, hn, ixb, H, qW, n);
          enc.set_function(_fn_qsa_store);
          enc.set_buffer(0, ixb); enc.set_buffer(1, xkp);
          enc.set_buffer(2, _pgtab);
          enc.set_constant(3, n_pages); enc.set_constant(4, page_tokens);
          enc.set_constant(5, Hkv); enc.set_constant(6, D);
          enc.set_constant(7, q_offset); enc.set_constant(8, qW);
          enc.set_constant(9, qh * qdim); enc.set_constant(10, xhead);
          enc.set_constant(11, xoff); enc.set_constant(12, qdim);
          enc.set_constant(13, n);
          enc.set_buffer(14, xvp); enc.set_buffer(15, qpos);
          enc.set_constant(16, ratio);
          enc.dispatch({(unsigned)qdim, (unsigned)n, 1},
                       {(unsigned)qdim, 1, 1});
          const int b0 = q_offset / ratio, b1 = (q_offset + n) / ratio;
          if (b1 > b0) {
            enc.set_function(_fn_qsa_pool);
            enc.set_buffer(0, xkp); enc.set_buffer(1, xvp);
            enc.set_buffer(2, _pgtab); enc.set_buffer(3, ly.ix_kn);
            enc.set_buffer(4, _inv_freq);
            enc.set_constant(5, n_pages); enc.set_constant(6, page_tokens);
            enc.set_constant(7, Hkv); enc.set_constant(8, D);
            enc.set_constant(9, xhead); enc.set_constant(10, xoff);
            enc.set_constant(11, qdim); enc.set_constant(12, ratio);
            enc.set_constant(13, b0); enc.set_constant(14, c.rotary_dim);
            enc.set_buffer(15, _hc_axis); enc.set_constant(16, eps);
            enc.dispatch({(unsigned)(qdim * (b1 - b0)), 1, 1},
                         {(unsigned)qdim, 1, 1});
          }
        }
        // A piece crossing into the sparse rows: its first r_s rows
        // attend densely on compact copies [Hq, r_s, D], the rest by QSA
        // below (the dense pass ran over all n and QSA overwrote n - r_s:
        // 180 ms of a 2048-row piece on an M5 Ultra).
        const int nd = part_dense ? r_s : n;
        const SharedBuffer& qx = part_dense ? qtd : qt;
        const SharedBuffer& ax = part_dense ? atd : at;
        if (part_dense) { hslice(qt, qtd, Hq, n * D, r_s * D, 0); }
        if (r_s > 0 && (part_dense ? prefill_set_serves_(nd) : use_pset)) {
          PrefillGqaAttnSet::Attn pa;
          pa.qt = &qx; pa.kpool = &kp; pa.vpool = &vp; pa.out = &ax;
          pa.page_table = &_pgtab;
          pa.n = nd; pa.q_offset = q_offset; pa.page_tokens = page_tokens;
          pa.n_pages = n_pages; pa.scale = scale;
          _prefill_set.dispatch(enc, pa);
        } else if (r_s > 0) {
          const bool flash = _flash_attn && _fn_sdpa_paged_flash.valid() &&
                             nd >= 384;
          const bool qtile = !flash && nd >= 384;
          // A decode step's row, a verify's few: each row's keys split
          // across 16 simdgroups (one walked them all, ~0.8 us a key a
          // layer), the split a function of the row's key count alone.
          const bool split = !flash && !qtile && nd <= 8 && D <= 256 &&
                             _fn_sdpa_paged_split.valid();
          enc.set_function(flash ? _fn_sdpa_paged_flash
                           : qtile ? _fn_sdpa_paged_qtile
                           : split ? _fn_sdpa_paged_split : _fn_sdpa_paged);
          enc.set_buffer(0, qx); enc.set_buffer(1, kp);
          enc.set_buffer(2, vp); enc.set_buffer(3, ax);
          enc.set_constant(4, scale); enc.set_constant(5, D);
          enc.set_constant(6, Hq); enc.set_constant(7, Hkv);
          enc.set_constant(8, nd); enc.set_constant(9, q_offset);
          enc.set_constant(10, page_tokens); enc.set_constant(11, n_pages);
          enc.set_buffer(12, _pgtab);
          if (flash) {
            enc.dispatch({256, (unsigned)Hq, (unsigned)((nd + 7) / 8)},
                         {256, 1, 1});
          } else if (qtile) {
            enc.dispatch({512, (unsigned)Hq, (unsigned)((nd + 15) / 16)},
                         {512, 1, 1});
          } else if (split) {
            enc.dispatch({32, 16u * (unsigned)Hq, (unsigned)nd},
                         {32, 16, 1});
          } else {
            enc.dispatch({32, (unsigned)Hq, (unsigned)nd}, {32, 1, 1});
          }
        }
        if (qsa && r_s < n) {
          // The rows that see more than budget + ratio - 1 tokens attend
          // only the top blocks the indexer picks (and their tail),
          // overwriting what the dense pass wrote for them.
          const int f = full_ord[(std::size_t)L];
          const SharedBuffer& xvp = *cx.vpool(nL + f / 4);
          const int xhead = (f % 4) / 2, xoff = ((f % 4) % 2) * qdim;
          enc.set_function(_fn_qsa_query);
          enc.set_buffer(0, ixb); enc.set_buffer(1, ixq);
          enc.set_buffer(2, ly.ix_qn); enc.set_buffer(3, _inv_freq);
          enc.set_constant(4, qW); enc.set_constant(5, qdim);
          enc.set_constant(6, qh); enc.set_constant(7, c.rotary_dim);
          enc.set_buffer(8, qpos); enc.set_constant(9, eps);
          enc.set_buffer(10, _hc_axis);
          enc.dispatch({(unsigned)qdim, (unsigned)qh, (unsigned)n},
                       {(unsigned)qdim, 1, 1});
          for (int r0 = r_s; r0 < n; r0 += qrb) {
            const int rows = std::min(qrb, n - r0);
            const int nbm = (q_offset + r0 + rows) / ratio;
            // A prefill's rows: the scores as matrix tiles; a decode
            // step's and a verify's a simdgroup a (block, row), the same
            // sums for both (VPIPE_QWEN_QSA_SSIMD=0 keeps a thread each:
            // an A/B).
            static const bool kScoreSimd = [] {
              const char* e = std::getenv("VPIPE_QWEN_QSA_SSIMD");
              return e == nullptr || std::atoi(e) != 0;
            }();
            const bool smma = n > 8 && qdim <= 128 && qdim % 8 == 0 &&
                              _fn_qsa_score_mma.valid();
            const bool ssimd = !smma && kScoreSimd && n <= 8 &&
                               qdim <= 128 && qdim % 32 == 0 &&
                               _fn_qsa_score_simd.valid();
            enc.set_function(smma    ? _fn_qsa_score_mma
                             : ssimd ? _fn_qsa_score_simd
                                     : _fn_qsa_score);
            enc.set_buffer(0, ixq); enc.set_buffer(1, xvp);
            enc.set_buffer(2, _pgtab); enc.set_buffer(3, qs_sc);
            enc.set_constant(4, n_pages); enc.set_constant(5, page_tokens);
            enc.set_constant(6, Hkv); enc.set_constant(7, D);
            enc.set_constant(8, xhead); enc.set_constant(9, xoff);
            enc.set_constant(10, qdim); enc.set_constant(11, qh);
            enc.set_constant(12, ratio); enc.set_constant(13, q_offset);
            enc.set_constant(14, r0); enc.set_constant(15, nbm);
            if (smma) {
              enc.set_constant(16, rows);
              enc.dispatch({128, (unsigned)((nbm + 31) / 32),
                            (unsigned)((rows + 31) / 32)}, {128, 1, 1});
            } else if (ssimd) {
              enc.dispatch({32u * (unsigned)nbm, (unsigned)rows, 1},
                           {256, 1, 1});
            } else {
              enc.dispatch({(unsigned)nbm, (unsigned)rows, 1}, {256, 1, 1});
            }
            enc.set_function(_fn_qsa_select);
            enc.set_buffer(0, qs_sc); enc.set_buffer(1, qs_sel);
            enc.set_buffer(2, qs_nsel);
            enc.set_constant(3, nbm); enc.set_constant(4, kblk);
            enc.set_constant(5, ratio); enc.set_constant(6, q_offset);
            enc.set_constant(7, r0);
            enc.dispatch({256u * (unsigned)rows, 1, 1}, {256, 1, 1});
            // A threadgroup a (head, row): one simdgroup's walk over the
            // 2051 selected keys took 5.8 ms a layer on an M5 Ultra, this
            // 0.28 (a row's sums its own: a verify's rows are a decode's).
            // A prefill's rows: a kv head's q-heads at once, as tiles --
            // the matrix units' matmul on an M5 (745 -> 113 ms a 2048-row
            // piece on the Ultra), else simdgroup MMA (-> ~240).
            const int G = Hq / Hkv;
            const bool wide = n > 8 && D == 256 && G <= 16;
            const bool qnax = wide && _use_mma && _fn_qsa_attn_nax.valid();
            const bool qmma = !qnax && wide && G >= 8 &&
                              _fn_qsa_attn_mma.valid();
            const bool qsplit = !qnax && !qmma && D <= 256 &&
                                _fn_qsa_attn_split.valid();
            if (!qnax && !qmma && qpart) {
              enc.set_function(_fn_qsa_attn_part);
              enc.set_buffer(0, qt); enc.set_buffer(1, kp);
              enc.set_buffer(2, vp); enc.set_buffer(3, qs_part);
              enc.set_buffer(4, _pgtab); enc.set_buffer(5, qs_sel);
              enc.set_buffer(6, qs_nsel);
              enc.set_constant(7, scale); enc.set_constant(8, D);
              enc.set_constant(9, Hq); enc.set_constant(10, Hkv);
              enc.set_constant(11, n); enc.set_constant(12, page_tokens);
              enc.set_constant(13, n_pages); enc.set_constant(14, sel_w);
              enc.set_constant(15, r0); enc.set_constant(16, qs_nch);
              enc.dispatch({128, (unsigned)Hq, (unsigned)(rows * qs_nch)},
                           {128, 1, 1});
              enc.set_function(_fn_qsa_attn_comb);
              enc.set_buffer(0, qs_part); enc.set_buffer(1, at);
              enc.set_buffer(2, qs_nsel);
              enc.set_constant(3, D); enc.set_constant(4, Hq);
              enc.set_constant(5, n); enc.set_constant(6, r0);
              enc.set_constant(7, qs_nch);
              enc.dispatch({256, (unsigned)Hq, (unsigned)rows}, {256, 1, 1});
            } else {
            enc.set_function(qnax ? _fn_qsa_attn_nax
                             : qmma ? _fn_qsa_attn_mma
                             : qsplit ? _fn_qsa_attn_split : _fn_qsa_attn);
            enc.set_buffer(0, qt); enc.set_buffer(1, kp);
            enc.set_buffer(2, vp); enc.set_buffer(3, at);
            enc.set_buffer(4, _pgtab); enc.set_buffer(5, qs_sel);
            enc.set_buffer(6, qs_nsel);
            enc.set_constant(7, scale); enc.set_constant(8, D);
            enc.set_constant(9, Hq); enc.set_constant(10, Hkv);
            enc.set_constant(11, n); enc.set_constant(12, page_tokens);
            enc.set_constant(13, n_pages); enc.set_constant(14, sel_w);
            enc.set_constant(15, r0);
            if (qnax || qmma) {
              enc.dispatch({256, (unsigned)Hkv, (unsigned)rows},
                           {256, 1, 1});
            } else if (qsplit) {
              enc.dispatch({32, 16u * (unsigned)Hq, (unsigned)rows},
                           {32, 16, 1});
            } else {
              enc.dispatch({32, (unsigned)Hq, (unsigned)rows}, {32, 1, 1});
            }
            }
            if (!dump_sel.empty() && r0 + rows == n) {
              // The last row's selection (+ its count) for the dump: its
              // int32s copied as pairs of 16-bit elements.
              enc.set_function(_fn_copy);
              enc.set_buffer(0, qs_sel,
                             (std::size_t)(rows - 1) * sel_w * 4);
              enc.set_buffer(1, dump_sel,
                             (std::size_t)f * (sel_w + 1) * 4);
              const int zero = 0, cnt = 2 * sel_w;
              enc.set_constant(2, zero); enc.set_constant(3, cnt);
              enc.dispatch({(unsigned)cnt, 1, 1}, {256, 1, 1});
              enc.set_buffer(0, qs_nsel, (std::size_t)(rows - 1) * 4);
              enc.set_buffer(1, dump_sel,
                             ((std::size_t)f * (sel_w + 1) + sel_w) * 4);
              const int two = 2;
              enc.set_constant(2, zero); enc.set_constant(3, two);
              enc.dispatch({2u, 1, 1}, {256, 1, 1});
            }
          }
        }
        if (af) {
          const int rd = part_dense ? r_s : 0;
          enc.set_function(_fn_hc_attn_post);
          enc.set_buffer(0, at); enc.set_buffer(1, part_dense ? atd : at);
          enc.set_buffer(2, qfull); enc.set_buffer(3, att);
          enc.set_constant(4, n); enc.set_constant(5, Hq);
          enc.set_constant(6, D); enc.set_constant(7, Nfqkv);
          enc.set_constant(8, rd);
          enc.dispatch({(unsigned)(n * qd), 1, 1}, {256, 1, 1});
        } else {
          transpose(at, att, Hq, n);
          if (part_dense) { transpose(atd, att, Hq, r_s); }
          enc.set_function(_fn_mul_sigmoid);
          enc.set_buffer(0, att); enc.set_buffer(1, gate3);
          enc.set_buffer(2, att); enc.set_constant(3, n * qd);
          enc.dispatch({(unsigned)(n * qd), 1, 1}, {256, 1, 1});
        }
        mm(ly.ow, ly.os, ly.ob, att, y, qd, H, n);
      } else {
        const SharedBuffer* csb = conv_in(L);
        const SharedBuffer* csw = conv_out(L);
        const SharedBuffer* ssb = ssm_in(L);
        const SharedBuffer* ssw = ssm_out(L);
        mm(ly.iqw, ly.iqs, ly.iqb, hn, mixqkv, H, Nf, n);
        // A decode step's and a verify's rows: the conv, the q / k norms
        // and g / beta in one launch, the step, its norm and gate in
        // another -- the arithmetic of the kernels below, z, a and b read
        // where the projection left them. VPIPE_QWEN_GDN_FUSED=0 keeps
        // those (an A/B), and is NOT bit-identical: a q / k norm reduces
        // in another order, a half-ulp apart in ~1 of 4k-50k values (M4
        // Pro, random inputs). A replay runs whichever the forward ran.
        static const bool kGdnFused = [] {
          const char* e = std::getenv("VPIPE_QWEN_GDN_FUSED");
          return e == nullptr || std::atoi(e) != 0;
        }();
        const bool gfused =
            kGdnFused && n <= 8 && Dk == 128 && Dv == 128 && Kc <= 9 &&
            keyd % 128 == 0 && Hv * Dv == vald && c.gdn_gate_sigmoid &&
            _fn_gdn_prework.valid() && _fn_gdn_step_ng.valid();
        // A verify keeps this layer's step inputs for a replay of the kept
        // prefix: the pre-conv qkv rows -- the whole in-projection rows
        // where the layer runs fused, which the replay then runs through
        // the same two kernels -- and g / beta in buffers of their own
        // (the shared ones are rewritten by the next layer).
        const SharedBuffer* lg = &gbuf;
        const SharedBuffer* lb = &betabuf;
        if (verify != nullptr) {
          GdnVerifyCache& gc = verify->gdn;
          const int W = gfused ? Nf : Cd;
          gc.layers.push_back(L);
          gc.full = gfused;
          gc.qkv.push_back(buf((std::size_t)n * W));
          gc.gbuf.push_back(f32b((std::size_t)n * Hv));
          gc.betabuf.push_back(f32b((std::size_t)n * Hv));
          hslice(mixqkv, gc.qkv.back(), n, Nf, W, 0);
          lg = &gc.gbuf.back();
          lb = &gc.betabuf.back();
        }
        if (gfused) {
          enc.set_function(_fn_gdn_prework);
          enc.set_buffer(0, *csb); enc.set_buffer(1, mixqkv);
          enc.set_buffer(2, ly.conv_w); enc.set_buffer(3, convout);
          enc.set_constant(4, n); enc.set_constant(5, Cd);
          enc.set_constant(6, Kc); enc.set_constant(7, Nf);
          enc.set_constant(8, keyd); enc.set_buffer(9, *csw);
          enc.set_buffer(10, _gdn_qscale); enc.set_buffer(11, _gdn_kscale);
          enc.set_buffer(12, ly.A_log); enc.set_buffer(13, ly.dt_bias);
          enc.set_buffer(14, *lg); enc.set_buffer(15, *lb);
          enc.set_constant(16, Hv); enc.set_constant(17, eps);
          enc.dispatch({(unsigned)(128 * (2 * keyd / 128 + Hv)), 1, 1},
                       {128, 1, 1});
          const std::size_t kb = (std::size_t)n * keyd * 2;
          enc.set_function(_fn_gdn_step_ng);
          enc.set_buffer(0, convout, 0); enc.set_buffer(1, convout, kb);
          enc.set_buffer(2, convout, 2 * kb); enc.set_buffer(3, *lg);
          enc.set_buffer(4, *lb); enc.set_buffer(5, *ssb);
          enc.set_buffer(7, *ssw);
          enc.set_constant(8, n); enc.set_constant(9, Hk);
          enc.set_constant(10, Hv); enc.set_buffer(11, ly.gdn_norm);
          enc.set_buffer(12, mixqkv); enc.set_constant(13, Nf);
          enc.set_constant(14, Cd); enc.set_constant(15, eps);
          enc.set_buffer(16, normout);
          enc.dispatch({32, 32, (unsigned)Hv}, {32, 32, 1});
          mm(ly.gow, ly.gos, ly.gob, normout, y, vald, H, n);
        } else {
          hslice(mixqkv, zbuf, n, Nf, vald, Cd);
          hslice(mixqkv, abuf, n, Nf, Hv, Cd + vald);
          hslice(mixqkv, bbuf, n, Nf, Hv, Cd + vald + Hv);
          // A prefill: a thread a (token, channel), then the state tail (the
          // kernel a channel a thread walked the n tokens alone: 54 ms of a
          // 2048-row piece). Each output the same sum, as compiled may round.
          const bool crows = n > 8 && _fn_gdn_conv1d_rows.valid() &&
                             _fn_gdn_conv1d_tail.valid();
          enc.set_function(crows ? _fn_gdn_conv1d_rows : _fn_gdn_conv1d);
          enc.set_buffer(0, *csb); enc.set_buffer(1, mixqkv);
          enc.set_buffer(2, ly.conv_w); enc.set_buffer(3, convout);
          enc.set_constant(4, n); enc.set_constant(5, Cd);
          enc.set_constant(6, Kc); enc.set_constant(7, Nf);
          enc.set_constant(8, keyd); enc.set_buffer(9, *csw);
          if (crows) {
            enc.dispatch({(unsigned)Cd, (unsigned)n, 1}, {256, 1, 1});
            // Its own bindings: a set_function may split the encoder, and a
            // fresh one carries none over (the tail then wrote no state).
            enc.set_function(_fn_gdn_conv1d_tail);
            enc.set_buffer(0, *csb); enc.set_buffer(1, mixqkv);
            enc.set_constant(4, n); enc.set_constant(5, Cd);
            enc.set_constant(6, Kc); enc.set_constant(7, Nf);
            enc.set_buffer(9, *csw);
            enc.dispatch({(unsigned)Cd, 1, 1}, {256, 1, 1});
          } else {
            enc.dispatch({(unsigned)Cd, 1, 1}, {256, 1, 1});
          }
          const std::size_t kb_off = (std::size_t)n * keyd * 2;
          const std::size_t vb_off = 2 * kb_off;
          rms(convout, 0, _gdn_qscale, convout, 0, n * Hk, Dk);
          rms(convout, kb_off, _gdn_kscale, convout, kb_off, n * Hk, Dk);
          enc.set_function(_fn_gdn_g_beta);
          enc.set_buffer(0, abuf); enc.set_buffer(1, bbuf);
          enc.set_buffer(2, ly.A_log); enc.set_buffer(3, ly.dt_bias);
          enc.set_buffer(4, *lg); enc.set_buffer(5, *lb);
          enc.set_constant(6, Hv); enc.set_constant(7, n);
          enc.dispatch({(unsigned)(n * Hv), 1, 1}, {256, 1, 1});
          const bool gdn4 = _fn_gdn_step_ndv4.valid() && (Dv % 4 == 0) &&
                            !_gdn_force_v1;
          enc.set_function(gdn4 ? _fn_gdn_step_ndv4 : _fn_gdn_step);
          enc.set_buffer(0, convout, 0); enc.set_buffer(1, convout, kb_off);
          enc.set_buffer(2, convout, vb_off); enc.set_buffer(3, *lg);
          enc.set_buffer(4, *lb); enc.set_buffer(5, *ssb);
          enc.set_buffer(6, ygdn); enc.set_buffer(7, *ssw);
          enc.set_constant(8, n); enc.set_constant(9, Hk);
          enc.set_constant(10, Hv);
          enc.dispatch({32, gdn4 ? (unsigned)(Dv / 4) : (unsigned)Dv,
                        (unsigned)Hv}, {32, 4, 1});
          rms(ygdn, 0, ly.gdn_norm, normout, 0, n * Hv, Dv);
          // The gate: sigmoid (output_gate_type) or silu (Qwen3.5's).
          if (c.gdn_gate_sigmoid) {
            enc.set_function(_fn_mul_sigmoid);
            enc.set_buffer(0, normout); enc.set_buffer(1, zbuf);
          } else {
            enc.set_function(_fn_swiglu);
            enc.set_buffer(0, zbuf); enc.set_buffer(1, normout);
          }
          enc.set_buffer(2, normout); enc.set_constant(3, n * vald);
          enc.dispatch({(unsigned)(n * vald), 1, 1}, {256, 1, 1});
          mm(ly.gow, ly.gos, ly.gob, normout, y, vald, H, n);
        }
      }
      hc_post(y);
      hc_pre(ly.hc_mlp, hn);
      MoeGrouped grp;
      if (grouped) {
        grp.hist = &g_hist; grp.boff = &g_boff; grp.curs = &g_curs;
        grp.t2e = &g_t2e; grp.ntile = &g_ntile; grp.srow = &g_srow;
        grp.sdst = &g_sdst; grp.gact = &g_gact; grp.gdout = &g_gdout;
        grp.npad = npad; grp.maxtiles = maxtiles; grp.maxm = maxm;
        grp.steel = steel;
      }
      encode_moe_mlp_(enc, ly, n, y, hn, moe_logits, moe_eid, moe_w, moe_act,
                      moe_part, moe_out, moe_ssg, moe_sout, moe_gate,
                      grouped ? &grp : nullptr);
      hc_post(y);
      // The DFlash drafter's taps: this layer's output, stream-averaged.
      if (run.tap_layers != nullptr && !mtp) {
        const std::vector<int>& tl = *run.tap_layers;
        const int trows = n - run.tap_row0;
        for (std::size_t i = 0; i < tl.size(); ++i) {
          if (tl[i] != L || trows <= 0) { continue; }
          hc_flush();
          enc.set_function(_fn_hc_stream_mean);
          enc.set_buffer(0, X);
          enc.set_buffer(1, *run.taps, i * (std::size_t)trows * H * 2);
          enc.set_constant(2, H); enc.set_constant(3, S);
          enc.set_constant(4, run.tap_row0);
          enc.dispatch({(unsigned)H, (unsigned)trows, 1}, {256, 1, 1});
        }
      }
      for (std::size_t i = 0; i < dump_layers.size(); ++i) {
        if (dump_layers[i] != L) { continue; }
        hc_flush();
        enc.set_function(_fn_copy);
        enc.set_buffer(0, X); enc.set_buffer(1, dumps[i]);
        const int zero = 0, cnt = n * SH;
        enc.set_constant(2, zero); enc.set_constant(3, cnt);
        enc.dispatch({(unsigned)cnt, 1, 1}, {256, 1, 1});
      }
    }
    if (run.resid != nullptr || run.no_head) { hc_flush(); }
    if (run.resid != nullptr) {
      // The residual streams the MTP head reads (before the final mixer).
      if (run.resid->empty() ||
          run.resid->byte_size() < (std::size_t)n * SH * 2) {
        *run.resid = buf((std::size_t)n * SH);
      }
      enc.set_function(_fn_copy);
      enc.set_buffer(0, X); enc.set_buffer(1, *run.resid);
      const int zero = 0, cnt = n * SH;
      enc.set_constant(2, zero); enc.set_constant(3, cnt);
      enc.dispatch({(unsigned)cnt, 1, 1}, {256, 1, 1});
    }
    if (!run.no_head) {
      // The final mixer, then the head over the rows asked for.
      hc_pre(mixer, hn);
      const int row0 = n - n_out;
      if (n_out == 1 && _lm_s.empty()) {
        dense_gemv_(enc, _lm_w, hn, (std::size_t)row0 * H * 2, _hc_logits, 0,
                    H, c.vocab);
      } else if (n_out == 1) {
        enc.set_function(_fn_qmv);
        enc.set_buffer(0, _lm_w); enc.set_buffer(1, _lm_s);
        enc.set_buffer(2, _lm_b);
        enc.set_buffer(3, hn, (std::size_t)row0 * H * 2);
        enc.set_buffer(4, _hc_logits);
        enc.set_constant(5, H); enc.set_constant(6, c.vocab);
        enc.dispatch({32, (unsigned)(c.vocab / 4), 1}, {32, 2, 1});
      } else {
        mm(_lm_w, _lm_s, _lm_b, hn, _hc_logits, H, c.vocab, n);
      }
      if (run.pick != nullptr) {
        if (_hc_pick.empty()) { _hc_pick = i32n(1); }
        encode_argmax_(enc, _hc_logits,
                       (std::size_t)(n_out - 1) * c.vocab * 2, _hc_pick, 0,
                       c.vocab);
      }
      if (run.tail) { run.tail(enc); }
    }
  }
  const auto tp2 = Clock::now();
  const bool done_ok = stream.commit().wait_ok();
  if (kProf && n == 1 && !mtp) {
    const auto tp3 = Clock::now();
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    _hc_prof[0] += ms(tp0, tp1);
    _hc_prof[1] += ms(tp1, tp2);
    _hc_prof[2] += ms(tp2, tp3);
    if (++_hc_prof_n % 16 == 0 && sess != nullptr) {
      const double k = 1.0 / _hc_prof_n;
      sess->info(fmt("[qwen] hc step profile ({} steps): ple + embed {:.2f} "
                     "ms, encode (with the cache's GPU waits) {:.2f} ms, "
                     "final wait {:.2f} ms", _hc_prof_n, _hc_prof[0] * k,
                     _hc_prof[1] * k, _hc_prof[2] * k));
    }
  }
  if (!done_ok) {
    if (sess != nullptr) {
      sess->warn(fmt("[qwen] Qwen3.8-Flash-Next: the forward's command "
                     "buffer failed"));
    }
    return {};
  }
  if (!dump_sel.empty()) {
    const std::string path = kSelDump != nullptr
                                 ? std::string(kSelDump)
                                 : std::string(kDump) + "/qsa_sel.i32";
    if (FILE* f = std::fopen(path.c_str(), "wb")) {
      std::fwrite(dump_sel.contents(), 1, dump_sel.byte_size(), f);
      std::fclose(f);
    }
  }
  for (std::size_t i = 0; i < dump_layers.size(); ++i) {
    const std::string path = std::string(kDump) + "/resid_L" +
                             std::to_string(dump_layers[i]) + ".f32";
    if (FILE* f = std::fopen(path.c_str(), "wb")) {
      const auto* d = static_cast<const std::uint16_t*>(dumps[i].contents());
      std::vector<float> row((std::size_t)SH);
      for (int r = 0; r < n; ++r) {
        for (int j = 0; j < SH; ++j) {
          row[(std::size_t)j] =
              elt_f32_(d[(std::size_t)r * SH + (std::size_t)j], c.use_bf16);
        }
        std::fwrite(row.data(), 4, row.size(), f);
      }
      std::fclose(f);
    }
  }
  // Nothing read back: a verify's rows stay in _hc_logits (its picks the
  // tail's) -- the last row's 248320 converted were never read.
  if (run.no_head || (run.all_rows && all_logits == nullptr)) {
    return std::vector<float>(1, 0.0f);
  }
  if (run.pick != nullptr) {
    *run.pick = static_cast<const std::int32_t*>(_hc_pick.contents())[0];
    return std::vector<float>(1, 0.0f);
  }
  std::vector<float> last((std::size_t)c.vocab);
  const auto* lg = static_cast<const std::uint16_t*>(_hc_logits.contents());
  const std::size_t lrow = (std::size_t)(n_out - 1) * c.vocab;
  for (int v = 0; v < c.vocab; ++v) {
    last[(std::size_t)v] = elt_f32_(lg[lrow + (std::size_t)v], c.use_bf16);
  }
  if (all_logits != nullptr) {
    all_logits->resize((std::size_t)n * c.vocab);
    for (std::size_t i = 0; i < (std::size_t)n * c.vocab; ++i) {
      (*all_logits)[i] = elt_f32_(lg[i], c.use_bf16);
    }
  }
  return last;
}

// ---- Lightning MTP speculative decode ------------------------------------

namespace {

// CONTEXT COPY: the context's tokens -- the prompt, then what was kept --
// and where each 6-gram of them ended, so a round whose tail occurred
// before drafts the tokens that followed it there (a reply quoting,
// rewriting or re-emitting what it was given) in place of the head's one.
// The match rule is MTPLX's: a 6-gram, the longest run back of up to 10
// among its last 32 occurrences, the most recent on a tie.
class CopyIndex {
public:
  static constexpr int kN = 6, kNmax = 10, kCands = 32;
  std::vector<std::int32_t> h;

  // Every 6-gram ending by h's end into the index.
  void sync() {
    for (std::size_t e = std::max<std::size_t>(_indexed + 1, kN);
         e <= h.size(); ++e) {
      _grams[key_(e)].push_back((int)e);
    }
    _indexed = h.size();
  }
  // Where the continuation of h's tail starts in an earlier occurrence,
  // and how far back that occurrence matches past the 6; -1 none.
  int find(int* ext_out) const {
    const int L = (int)h.size();
    if (L < kN + 1) { return -1; }
    const auto it = _grams.find(key_((std::size_t)L));
    if (it == _grams.end()) { return -1; }
    const std::vector<int>& c = it->second;
    int best = -1, best_ext = -1;
    const int lo = std::max(0, (int)c.size() - kCands);
    for (int i = (int)c.size() - 1; i >= lo; --i) {
      const int pos = c[(std::size_t)i];
      if (pos >= L) { continue; }                     // the tail itself
      if (!std::equal(h.begin() + (pos - kN), h.begin() + pos,
                      h.begin() + (L - kN))) {
        continue;                                     // a hash collision
      }
      int ext = 0;
      while (ext < kNmax - kN && pos - kN - 1 - ext >= 0 &&
             h[(std::size_t)(pos - kN - 1 - ext)] ==
                 h[(std::size_t)(L - kN - 1 - ext)]) {
        ++ext;
      }
      if (ext > best_ext) {
        best_ext = ext;
        best = pos;
        if (ext == kNmax - kN) { break; }
      }
    }
    if (ext_out != nullptr) { *ext_out = best_ext; }
    return best;
  }

private:
  std::uint64_t key_(std::size_t e) const {
    std::uint64_t k = 1469598103934665603ull;
    for (std::size_t i = e - kN; i < e; ++i) {
      k = (k ^ (std::uint64_t)(std::uint32_t)h[i]) * 1099511628211ull;
    }
    return k;
  }
  std::unordered_map<std::uint64_t, std::vector<int>> _grams;
  std::size_t _indexed = 0;
};

}  // namespace

void
MetalQwenModel::encode_hc_replay_(ComputeEncoder& enc, ContextId cid,
                                  int keep, const HcVerify& hv,
                                  std::span<const std::int32_t> ids)
{
  if (keep <= 0) { return; }
  const Config& c = _cfg;
  const int Cd = c.gdn_conv_dim, keyd = c.key_dim(), vald = c.value_dim();
  const int Hv = c.gdn_v_heads, Dv = c.gdn_v_dim, Dk = c.gdn_k_dim;
  const int Hk = c.gdn_k_heads, Kc = c.gdn_conv_kernel;
  const float eps = c.rms_eps;
  const GdnVerifyCache& gc = hv.gdn;
  const SharedBuffer& convout =
      verify_scratch_(40, (std::size_t)keep * Cd * 2);
  const SharedBuffer& ygdn = verify_scratch_(41, (std::size_t)keep * vald * 2);
  const std::size_t kb_off = (std::size_t)keep * keyd * 2;
  const std::size_t vb_off = 2 * kb_off;
  const bool gdn4 = _fn_gdn_step_ndv4.valid() && (Dv % 4 == 0) &&
                    !_gdn_force_v1;
  auto rms = [&](std::size_t off, const SharedBuffer& w, int rows) {
    enc.set_function(rms_fn_(Dk, off, off));
    enc.set_buffer(0, convout, off); enc.set_buffer(1, w);
    enc.set_buffer(2, convout, off);
    enc.set_constant(3, Dk); enc.set_constant(4, eps);
    enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
  };
  for (std::size_t gi = 0; gi < gc.layers.size(); ++gi) {
    const int L = gc.layers[gi];
    const Layer& ly = _layers[(std::size_t)L];
    const SharedBuffer* cr = _ctx->conv_slot(cid, L, 0);
    const SharedBuffer* cw = _ctx->conv_slot(cid, L, 1);
    const SharedBuffer* sr = _ctx->ssm_slot(cid, L, 0);
    const SharedBuffer* sw = _ctx->ssm_slot(cid, L, 1);
    if (!cr || !cw || !sr || !sw) { continue; }
    if (gc.full) {
      // The forward ran this layer fused: the kept rows go through the
      // SAME two kernels, bound as forward_hc_ binds them, on the same
      // in-projection rows -- the decode's arithmetic by construction.
      // (conv + rms_norm_fast + ndv4 compute the same values only by
      // reading alike: on random inputs a q / k norm came out a half-ulp
      // apart about once in 10k-50k values on an M4 Pro, though none
      // showed in 20 replays of the real model.) g / beta are rewritten
      // with the values the verify wrote there, from the same a / b; the
      // gated output is not wanted and lands in `ygdn`.
      const int Nf = Cd + vald + 2 * Hv;
      enc.set_function(_fn_gdn_prework);
      enc.set_buffer(0, *cr); enc.set_buffer(1, gc.qkv[gi]);
      enc.set_buffer(2, ly.conv_w); enc.set_buffer(3, convout);
      enc.set_constant(4, keep); enc.set_constant(5, Cd);
      enc.set_constant(6, Kc); enc.set_constant(7, Nf);
      enc.set_constant(8, keyd); enc.set_buffer(9, *cw);
      enc.set_buffer(10, _gdn_qscale); enc.set_buffer(11, _gdn_kscale);
      enc.set_buffer(12, ly.A_log); enc.set_buffer(13, ly.dt_bias);
      enc.set_buffer(14, gc.gbuf[gi]); enc.set_buffer(15, gc.betabuf[gi]);
      enc.set_constant(16, Hv); enc.set_constant(17, eps);
      enc.dispatch({(unsigned)(128 * (2 * keyd / 128 + Hv)), 1, 1},
                   {128, 1, 1});
      enc.set_function(_fn_gdn_step_ng);
      enc.set_buffer(0, convout, 0); enc.set_buffer(1, convout, kb_off);
      enc.set_buffer(2, convout, vb_off); enc.set_buffer(3, gc.gbuf[gi]);
      enc.set_buffer(4, gc.betabuf[gi]); enc.set_buffer(5, *sr);
      enc.set_buffer(7, *sw);
      enc.set_constant(8, keep); enc.set_constant(9, Hk);
      enc.set_constant(10, Hv); enc.set_buffer(11, ly.gdn_norm);
      enc.set_buffer(12, gc.qkv[gi]); enc.set_constant(13, Nf);
      enc.set_constant(14, Cd); enc.set_constant(15, eps);
      enc.set_buffer(16, ygdn);
      enc.dispatch({32, 32, (unsigned)Hv}, {32, 32, 1});
      continue;
    }
    enc.set_function(_fn_gdn_conv1d);
    enc.set_buffer(0, *cr); enc.set_buffer(1, gc.qkv[gi]);
    enc.set_buffer(2, ly.conv_w); enc.set_buffer(3, convout);
    enc.set_constant(4, keep); enc.set_constant(5, Cd);
    enc.set_constant(6, Kc); enc.set_constant(7, Cd);
    enc.set_constant(8, keyd); enc.set_buffer(9, *cw);
    enc.dispatch({(unsigned)Cd, 1, 1}, {256, 1, 1});
    rms(0, _gdn_qscale, keep * Hk);
    rms(kb_off, _gdn_kscale, keep * Hk);
    enc.set_function(gdn4 ? _fn_gdn_step_ndv4 : _fn_gdn_step);
    enc.set_buffer(0, convout, 0); enc.set_buffer(1, convout, kb_off);
    enc.set_buffer(2, convout, vb_off); enc.set_buffer(3, gc.gbuf[gi]);
    enc.set_buffer(4, gc.betabuf[gi]); enc.set_buffer(5, *sr);
    enc.set_buffer(6, ygdn); enc.set_buffer(7, *sw);
    enc.set_constant(8, keep); enc.set_constant(9, Hk);
    enc.set_constant(10, Hv);
    enc.dispatch({32, gdn4 ? (unsigned)(Dv / 4) : (unsigned)Dv,
                  (unsigned)Hv}, {32, 4, 1});
  }
  if (_ple == nullptr) { return; }
  const int Lp = c.ple_layer, SH = c.hc_count * c.hidden;
  const SharedBuffer* cs = _ctx->conv_slot(cid, Lp, 0);
  const SharedBuffer* co = _ctx->conv_slot(cid, Lp, 1);
  if (cs == nullptr || co == nullptr) { return; }
  enc.set_function(_fn_ple_state);
  enc.set_buffer(0, *cs, _ple->hist_off); enc.set_buffer(1, hv.gvn);
  enc.set_buffer(2, *co, _ple->hist_off);
  enc.set_constant(3, SH); enc.set_constant(4, keep);
  enc.set_constant(5, _ple->hist);
  enc.dispatch({(unsigned)SH, 1, 1}, {256, 1, 1});
  // The last two ids after `keep` rows (the verify wrote its n rows').
  const auto* st = reinterpret_cast<const std::int32_t*>(
      static_cast<const std::uint8_t*>(cs->contents()) + _ple->ids_off);
  auto* so = reinterpret_cast<std::int32_t*>(
      static_cast<std::uint8_t*>(co->contents()) + _ple->ids_off);
  const std::int64_t eos = c.ple_eos;
  std::int64_t h[2] = {st[0] > 0 ? st[0] - 1 : eos,
                       st[1] > 0 ? st[1] - 1 : eos};
  for (int t = 0; t < keep; ++t) {
    h[0] = h[1];
    h[1] = ids[(std::size_t)t];
  }
  so[0] = (std::int32_t)h[0] + 1;
  so[1] = (std::int32_t)h[1] + 1;
}

bool
MetalQwenModel::mtp_decode_hc_(ContextId cid, std::int32_t first_token,
                               int n_steps, std::vector<std::int32_t>& out_ids,
                               long* accepted_out, long* rounds_out,
                               const MtpDecodeCtl& ctl)
{
  out_ids.clear();
  if (ctl.hit_stop) { *ctl.hit_stop = false; }
  if (accepted_out) { *accepted_out = 0; }
  if (rounds_out) { *rounds_out = 0; }
  if (_hc_mtp == nullptr || n_steps <= 0) { return false; }
  const Config& c = _cfg;
  const std::size_t V = (std::size_t)c.vocab;
  const std::size_t row = (std::size_t)c.hc_count * c.hidden * 2;
  const GpuSamplerParams& sp = ctl.sampler;
  const bool sampling = !sp.greedy;
  if (!_fn_argmax.valid() || (sampling && !_fn_sample.valid())) {
    return false;
  }
  const bool penal = sampling && (sp.repetition_penalty != 1.0f ||
                                  sp.presence_penalty != 0.0f);
  // Drafts a round, the engine's: each one the head's single layer, a
  // fraction of a verify row. MEASURED on the M4 Pro (64 GB, the experts
  // cached), 128 greedy tokens, ms a token, cold cache / warm: plain 150 /
  // 52, depth 2 129 / 46, depth 3 116 / 44, depth 4 - / 58 (3.0 tokens a
  // round at 3 and at 4: a fifth row costs more than it keeps). Every
  // expert resident (an M5 Ultra), a verify's row costs a third of a step
  // and the head keeps two drafts in three: depth 1 107 tok/s, 2 99, 3 90
  // on a short prompt; 97.4 / 97.0 at 1 / 2 past a 7k one.
  // VPIPE_QWEN_HC_MTP_DEPTH pins it (0: none -- the verify then runs one
  // row, a plain decode step).
  int depth = _xc != nullptr ? 3 : 1;
  if (const char* e = std::getenv("VPIPE_QWEN_HC_MTP_DEPTH")) {
    depth = std::clamp(std::atoi(e), 0, 8);
  }
  HcMtp& M = *_hc_mtp;
  const int main0 = _ctx->seq_len_of(cid);
  HcMtp::Seq& sq = hc_mtp_seq_(cid, main0);
  // GDN double buffer as dflash_decode's: the verify reads ring slot 0 and
  // writes slot 1; a partial accept replays the kept rows 0 -> 1.
  if (!_ctx->gdn_ring_begin(cid, 2)) { return false; }
  // CONTEXT COPY (CopyIndex): a round whose tail occurred earlier in the
  // context verifies the tokens that followed it there in place of the
  // head's draft -- up to seven, a verify of eight rows, the most whose
  // sums are a decode step's -- the head folding what it owes all the
  // same. Its rounds are timed against the head's: while it costs more a
  // kept token it sits out 32 rounds (a wrong block of seven is a verify
  // of eight rows, 31.7 ms against a head round's 15 on an M5 Ultra), and
  // until it has paid in this decode its blocks are three drafts -- then
  // a match that runs further back earns a longer one. The output is the
  // verifier's picks either way. VPIPE_QWEN_HC_COPY=0 turns it off.
  static const bool kCopy = [] {
    const char* e = std::getenv("VPIPE_QWEN_HC_COPY");
    return e == nullptr || std::atoi(e) != 0;
  }();
  constexpr int kCopyMax = 7;
  const int cap = std::max(depth, kCopy ? kCopyMax : 0) + 1;
  SharedBuffer fold_in = _mc->make_shared_buffer((std::size_t)cap * row);
  SharedBuffer hin = _mc->make_shared_buffer(row);
  SharedBuffer fold_last = _mc->make_shared_buffer(row);
  SharedBuffer picks = _mc->make_shared_buffer((std::size_t)cap * 4);
  SharedBuffer ws = _mc->make_shared_buffer(V * 2);
  SharedBuffer seen_rows = _mc->make_shared_buffer((std::size_t)cap * V);
  if (fold_in.empty() || hin.empty() || fold_last.empty() || picks.empty() ||
      ws.empty() || seen_rows.empty()) {
    _ctx->gdn_ring_end(cid);
    return false;
  }
  std::memset(seen_rows.contents(), 0, seen_rows.byte_size());
  std::vector<std::uint8_t> seen_base;
  if (penal) {
    seen_base.assign(V, 0);
    for (const std::int32_t t : ctl.prime) {
      if (t >= 0 && (std::size_t)t < V) { seen_base[(std::size_t)t] = 1; }
    }
  }
  // The pairs owed to the head: residual rows and the tokens after them.
  // A head in step with this context owes (pending, first token).
  std::vector<std::int32_t> fold_ids;
  if (sq.main_len >= 0 && main0 > 0) {
    std::memcpy(fold_in.contents(), sq.pending.contents(), row);
    fold_ids.push_back(first_token);
  }
  SharedBuffer mres, vres;
  std::int32_t anchor = first_token;
  long accepted = 0, rounds = 0, drafted = 0;
  bool terminate = false, failed = false;
  int keep = 0;
  int folded = 0;                     // head rows this round's fold wrote
  std::vector<std::int32_t> vids, preds;
  CopyIndex ci;
  if (kCopy) {
    ci.h.assign(ctl.prime.begin(), ctl.prime.end());
    ci.h.push_back(first_token);
  }
  int copy_off = 0;                   // rounds the copy lane sits out
  bool copy_paid = false;             // its rounds have cost less a token
  int copy_seen = 0;                  // its rounds since it last sat out
  double copy_rate = -1.0, head_rate = -1.0;   // ms a kept token, EMAs
  long copy_rounds = 0, copy_drafted = 0, copy_kept = 0;
  using clk = std::chrono::steady_clock;
  static const bool kProf =
      std::getenv("VPIPE_QWEN_HC_MTP_PROFILE") != nullptr;
  double t_draft = 0, t_verify = 0, t_settle = 0;
  while (!terminate && (int)out_ids.size() < n_steps) {
    const auto t0 = clk::now();
    const int budget = n_steps - (int)out_ids.size();
    // The copy lane's block: 3 drafts until it has paid; then 3 for the
    // bare 6-gram, 5 for a match a token or two further back, 7 past.
    std::vector<std::int32_t> copy;
    if (kCopy && copy_off > 0) {
      --copy_off;
    } else if (kCopy && !fold_ids.empty() && budget > 2) {
      ci.sync();
      int ext = 0;
      const int at = ci.find(&ext);
      if (at >= 0) {
        const int want = !copy_paid ? 3 : ext >= 3 ? 7 : ext >= 1 ? 5 : 3;
        const int len = std::min({want, kCopyMax, budget - 1,
                                  (int)ci.h.size() - at});
        copy.assign(ci.h.begin() + at, ci.h.begin() + at + len);
      }
    }
    const int k = fold_ids.empty() || !copy.empty()
                      ? 0 : std::min(depth, budget - 1);
    vids.assign(1, anchor);
    // Fold what is owed; its last row's logits draft the first token, and
    // each draft chains on the head's own residual for the next. The fold
    // is history; the chain's rows are speculative and given back below.
    folded = 0;
    int head_len = M.ctx->seq_len_of(sq.cid);   // its history, after the fold
    if (!fold_ids.empty()) {
      const int nf = (int)fold_ids.size();
      std::memcpy(fold_last.contents(),
                  static_cast<const std::uint8_t*>(fold_in.contents()) +
                      (std::size_t)(nf - 1) * row, row);
      // The drafts' ids picked on the GPU (their logits never read back).
      std::int32_t d = 0;
      HcRun r;
      r.mtp_in = &fold_in;
      r.resid = k > 1 ? &mres : nullptr;
      r.no_head = k == 0;
      r.pick = &d;
      std::vector<float> lg = forward_hc_(sq.cid, fold_ids, nullptr, nullptr,
                                          nullptr, -1, nullptr, &r);
      if (lg.empty()) { failed = true; break; }
      folded = nf;
      fold_ids.clear();
      head_len = M.ctx->seq_len_of(sq.cid);
      if (k > 0) {
        vids.push_back(d);
        if (k > 1) {
          std::memcpy(hin.contents(),
                      static_cast<const std::uint8_t*>(mres.contents()) +
                          (std::size_t)(nf - 1) * row, row);
        }
        for (int i = 1; i < k; ++i) {
          HcRun r2;
          r2.mtp_in = &hin;
          r2.resid = &mres;
          std::int32_t d2 = 0;
          r2.pick = &d2;
          lg = forward_hc_(sq.cid, std::span<const std::int32_t>(&d, 1),
                           nullptr, nullptr, nullptr, -1, nullptr, &r2);
          if (lg.empty()) { break; }
          d = d2;
          vids.push_back(d);
          std::memcpy(hin.contents(), mres.contents(), row);
        }
      }
    }
    vids.insert(vids.end(), copy.begin(), copy.end());
    const auto t1 = clk::now();
    const int nv = (int)vids.size();
    const int pos = _ctx->seq_len_of(cid);
    const int g0 = (int)out_ids.size();     // the anchor's index in the run
    // After an image turn the rows take the mROPE-advanced positions:
    // rope_first for the first decoded token, on from there.
    const int rp = ctl.rope_first < 0 ? -1 : ctl.rope_first + (pos - main0);
    // The verifier's pick at every row, on the GPU as a decode step picks
    // -- argmax, or a sample seeded by the token's index in the run,
    // against the seen-set a step would hold there -- encoded at the
    // verify's end.
    if (penal) {
      auto* sr = static_cast<std::uint8_t*>(seen_rows.contents());
      for (int i = 0; i < nv; ++i) {
        std::uint8_t* rw = sr + (std::size_t)i * V;
        std::memcpy(rw, seen_base.data(), V);
        for (int q = 0; q <= i; ++q) {
          const std::int32_t t = vids[(std::size_t)q];
          if (t >= 0 && (std::size_t)t < V) { rw[(std::size_t)t] = 1; }
        }
      }
    }
    HcVerify hv;
    hv.ring = true;
    HcRun vr;
    vr.verify = &hv;
    vr.resid = &vres;
    vr.all_rows = true;
    vr.tail = [&](ComputeEncoder& enc) {
      for (int i = 0; i < nv; ++i) {
        const std::size_t lo = (std::size_t)i * V * 2;
        if (!sampling) {
          encode_argmax_(enc, _hc_logits, lo, picks, (std::size_t)i * 4,
                         c.vocab);
        } else {
          const std::uint32_t seed = (std::uint32_t)(
              sp.seed + 0x9e3779b9ull * (std::uint64_t)(g0 + i + 1));
          encode_sample_core_(enc, _hc_logits, lo, picks,
                              (std::size_t)i * 4, ws, 0, seen_rows,
                              (std::size_t)i * V, sp, seed, c.vocab);
        }
      }
    };
    const bool vok = !forward_hc_(cid, vids, nullptr, nullptr, nullptr, rp,
                                  nullptr, &vr).empty();
    // The chain's rows (and any a failed step left) go: the head keeps
    // only committed pairs.
    if (const int extra = M.ctx->seq_len_of(sq.cid) - head_len; extra > 0) {
      M.ctx->kv_rollback(sq.cid, extra);
    }
    if (!vok) {
      const int extra = _ctx->seq_len_of(cid) - pos;
      if (extra > 0) { _ctx->kv_rollback(cid, extra); }
      failed = true;
      break;
    }
    preds.assign(static_cast<const std::int32_t*>(picks.contents()),
                 static_cast<const std::int32_t*>(picks.contents()) + nv);
    ++rounds;
    // The longest prefix the verifier agrees with; the anchor always is.
    int j = 0;
    while (j < nv - 1 && vids[(std::size_t)j + 1] == preds[(std::size_t)j]) {
      ++j;
    }
    keep = j + 1;
    // A stop token ends the decode and is not kept (as mtp_decode).
    if (ctl.is_stop) {
      for (int i = 0; i < keep; ++i) {
        if (ctl.is_stop(vids[(std::size_t)i])) {
          keep = i;
          terminate = true;
          if (ctl.hit_stop) { *ctl.hit_stop = true; }
          break;
        }
      }
    }
    for (int i = 0; i < keep; ++i) {
      out_ids.push_back(vids[(std::size_t)i]);
      if (penal) {
        const std::int32_t t = vids[(std::size_t)i];
        if (t >= 0 && (std::size_t)t < V) { seen_base[(std::size_t)t] = 1; }
      }
    }
    drafted += nv - 1;
    if (keep > 1) { accepted += keep - 1; }
    if (!copy.empty()) {
      ++copy_rounds;
      copy_drafted += (long)copy.size();
      copy_kept += std::max(0, keep - 1);
    }
    const auto t2 = clk::now();
    // The main context: the rejected rows out of the KV; the GDN and PLE
    // state the kept rows' (slot 1 whole, else replayed into it).
    if (keep < nv) { _ctx->kv_rollback(cid, nv - keep); }
    if (keep == nv) {
      _ctx->gdn_ring_advance(cid);
    } else if (keep > 0) {
      metal_compute::CommandStream& st = spec_stream_();
      {
        ComputeEncoder enc = st.begin_compute();
        encode_hc_replay_(enc, cid, keep, hv, vids);
      }
      if (!st.commit().wait_ok()) { failed = true; break; }
      _ctx->gdn_ring_advance(cid);
    }
    if (ctl.on_round && keep > 0) {
      const std::span<const std::int32_t> fresh(
          out_ids.data() + (out_ids.size() - (std::size_t)keep),
          (std::size_t)keep);
      if (!ctl.on_round(fresh)) { terminate = true; }
    }
    if (keep == 0) { terminate = true; }
    if (!terminate && (int)out_ids.size() < n_steps) {
      // Owed to the next round: (row i, the token after it) for the kept
      // rows, the last paired with the verifier's next pick -- the anchor.
      std::memcpy(fold_in.contents(), vres.contents(), (std::size_t)keep * row);
      fold_ids.assign(vids.begin() + 1, vids.begin() + keep);
      fold_ids.push_back(preds[(std::size_t)keep - 1]);
      anchor = preds[(std::size_t)keep - 1];
      if (kCopy) {
        ci.h.insert(ci.h.end(), vids.begin() + 1, vids.begin() + keep);
        ci.h.push_back(anchor);
      }
    }
    const auto t3 = clk::now();
    // The copy lane against the head, a kept token's milliseconds.
    if (kCopy && keep > 0 && (!copy.empty() || k > 0)) {
      const double r =
          std::chrono::duration<double, std::milli>(t3 - t0).count() / keep;
      double& rate = !copy.empty() ? copy_rate : head_rate;
      rate = rate < 0 ? r : 0.7 * rate + 0.3 * r;
      if (!copy.empty() && head_rate > 0 && ++copy_seen >= 2) {
        if (copy_rate > head_rate) {
          copy_off = 32;
          copy_paid = false;
          copy_seen = 0;
          copy_rate = -1.0;
        } else {
          copy_paid = true;
        }
      }
    }
    if (kProf) {
      auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
      };
      t_draft += ms(t0, t1);
      t_verify += ms(t1, t2);
      t_settle += ms(t2, t3);
    }
  }
  // Leave the head in step with the context: every kept pair but the last
  // folded, the last row's residual pending.
  bool settled = !failed;
  if (settled && keep > 0) {
    if (keep > 1) {
      std::memcpy(fold_in.contents(), vres.contents(),
                  (std::size_t)(keep - 1) * row);
      HcRun r;
      r.mtp_in = &fold_in;
      r.no_head = true;
      const std::span<const std::int32_t> next(vids.data() + 1,
                                               (std::size_t)keep - 1);
      settled = !forward_hc_(sq.cid, next, nullptr, nullptr, nullptr, -1,
                             nullptr, &r).empty();
    }
    std::memcpy(sq.pending.contents(),
                static_cast<const std::uint8_t*>(vres.contents()) +
                    (std::size_t)(keep - 1) * row, row);
  } else if (settled && rounds > 0) {
    // Nothing kept (the anchor was a stop): its fold row goes, and the
    // residual it carried is pending again.
    if (folded > 0) {
      M.ctx->kv_rollback(sq.cid, 1);
      std::memcpy(sq.pending.contents(), fold_last.contents(), row);
    } else {
      settled = false;
    }
  }
  sq.main_len = settled ? _ctx->seq_len_of(cid) : -1;
  _ctx->gdn_ring_end(cid);
  if (accepted_out) { *accepted_out = accepted; }
  if (rounds_out) { *rounds_out = rounds; }
  if (ctl.drafted) { *ctl.drafted = drafted; }
  if (kProf && rounds > 0) {
    if (const SessionContextIntf* s = _mc->session()) {
      s->info(fmt("[qwen] MTP depth {}: {} tokens in {} rounds ({:.2f} a "
                  "round), {} of {} drafts kept; per round draft {:.1f} ms, "
                  "verify {:.1f} ms, settle {:.1f} ms; context copy {} "
                  "rounds, {} of {} drafts kept", depth,
                  out_ids.size(), rounds,
                  (double)out_ids.size() / (double)rounds, accepted, drafted,
                  t_draft / rounds, t_verify / rounds, t_settle / rounds,
                  copy_rounds, copy_kept, copy_drafted));
    }
  }
  return !failed || !out_ids.empty();
}

MetalQwenModel::HcTaps
MetalQwenModel::hc_taps_begin_(ContextId cid, int n, int piece)
{
  HcTaps t;
  if (_dflash == nullptr || n <= 0) { return t; }
  t.n = n;
  t.trows = dflash_tap_rows_(n);
  t.pos0 = _ctx->seq_len_of(cid);
  t.layers = &_dflash->config().target_layer_ids;
  if (t.trows > 0) {
    t.buf = _mc->make_shared_buffer(t.layers->size() *
                                    (std::size_t)std::min(piece, t.trows) *
                                    _cfg.hidden * 2);
    if (t.buf.empty()) { t.trows = 0; }
  }
  // A prefill at position 0 is a fresh context, whatever its id: ids are
  // recycled, and a recycled one must not see the last conversation.
  if (t.pos0 == 0) { _dflash_cid = {}; }
  dflash_bind_ctx_(cid);
  return t;
}

void
MetalQwenModel::hc_taps_piece_(const HcTaps& t, int i0, int m,
                               HcRun& run) const
{
  run.tap_layers = nullptr;
  run.taps = nullptr;
  run.tap_row0 = 0;
  const int r0 = std::max(0, t.n - t.trows - i0);  // first row it sees
  if (t.trows <= 0 || r0 >= m) { return; }
  run.tap_layers = t.layers;
  run.taps = &t.buf;
  run.tap_row0 = r0;
}

bool
MetalQwenModel::hc_taps_ingest_(const HcTaps& t, int i0, int m,
                                const HcRun& run)
{
  if (run.tap_layers == nullptr || _dflash == nullptr) { return true; }
  const int rows = m - run.tap_row0;
  metal_compute::CommandStream& st = spec_stream_();
  {
    ComputeEncoder enc = st.begin_compute();
    _dflash->encode_ingest(enc, t.buf, rows, 0, rows,
                           t.pos0 + i0 + run.tap_row0);
  }
  return st.commit().wait_ok();
}

bool
MetalQwenModel::dflash_decode_hc_(ContextId cid, std::int32_t first_token,
                                  int n_steps,
                                  std::vector<std::int32_t>& out_ids,
                                  const MtpDecodeCtl& ctl, int block,
                                  SpecStats* stats)
{
  const Config& c = _cfg;
  const DFlashConfig& dc = _dflash->config();
  DFlashTarget* hooks = dflash_target();
  if (hooks == nullptr || !_fn_hc_stream_mean.valid()) { return false; }
  const std::size_t V = (std::size_t)c.vocab;
  const int H = c.hidden;
  const GpuSamplerParams& sp = ctl.sampler;
  const bool sampling = !sp.greedy;
  if (!_fn_argmax.valid() || (sampling && !_fn_sample.valid())) {
    return false;
  }
  const bool penal = sampling && (sp.repetition_penalty != 1.0f ||
                                  sp.presence_penalty != 0.0f);
  // Block length: the anchor plus the drafts the verifier sees -- fixed
  // when asked for, else adapted per round (DFlashTuner). The verify's
  // rows are each computed as a decode step computes its one up to 8 (the
  // MoE router and shared expert a row at a time), which keeps the tokens
  // the plain decode's, so a round is held to 8.
  const bool adaptive = block <= 0;
  const int Lmax = std::clamp(dc.block_size, 2, 8);
  const int Lb = adaptive ? Lmax : std::clamp(block, 2, Lmax);
  const int main0 = _ctx->seq_len_of(cid);
  dflash_bind_ctx_(cid);
  // GDN double buffer as dflash_decode's: the verify reads ring slot 0 and
  // writes slot 1; a partial accept replays the kept rows 0 -> 1.
  if (!_ctx->gdn_ring_begin(cid, 2)) { return false; }
  const std::vector<int>& tl = dc.target_layer_ids;
  SharedBuffer taps =
      _mc->make_shared_buffer(tl.size() * (std::size_t)Lmax * H * 2);
  SharedBuffer drafts = _mc->make_shared_buffer((std::size_t)Lmax * 4);
  SharedBuffer blk = _mc->make_shared_buffer((std::size_t)Lmax * 4);
  SharedBuffer picks = _mc->make_shared_buffer((std::size_t)Lmax * 4);
  SharedBuffer ws = _mc->make_shared_buffer(V * 2);
  SharedBuffer seen_rows = _mc->make_shared_buffer((std::size_t)Lmax * V);
  if (taps.empty() || drafts.empty() || blk.empty() || picks.empty() ||
      ws.empty() || seen_rows.empty()) {
    _ctx->gdn_ring_end(cid);
    return false;
  }
  std::memset(seen_rows.contents(), 0, seen_rows.byte_size());
  auto* dp = static_cast<std::int32_t*>(drafts.contents());
  auto* bp = static_cast<std::int32_t*>(blk.contents());
  std::vector<std::uint8_t> seen_base;
  if (penal) {
    seen_base.assign(V, 0);
    for (const std::int32_t t : ctl.prime) {
      if (t >= 0 && (std::size_t)t < V) { seen_base[(std::size_t)t] = 1; }
    }
  }
  // What a round leaves owed to the next one's first buffer: the GDN /
  // PLE replay of a partial accept (over the verify that made it), and
  // the kept rows' taps for the drafter.
  HcVerify hv_owed;
  std::vector<std::int32_t> vids_owed;
  int owe_keep = 0, owe_rows = 0, owe_pos = 0, owe_L = 0;
  auto encode_owed = [&](ComputeEncoder& enc) {
    if (owe_keep > 0) {
      encode_hc_replay_(enc, cid, owe_keep, hv_owed, vids_owed);
      _ctx->gdn_ring_advance(cid);
      owe_keep = 0;
    }
    if (owe_rows > 0) {
      _dflash->encode_ingest(enc, taps, owe_L, 0, owe_rows, owe_pos);
      owe_rows = 0;
    }
  };
  using clk = std::chrono::steady_clock;
  static const bool kProf = std::getenv("VPIPE_DFLASH_PROFILE") != nullptr;
  double t_draft = 0.0, t_verify = 0.0;
  SpecStats st;
  std::int32_t anchor = first_token;
  bool terminate = false, failed = false;
  std::vector<std::int32_t> vids, preds;
  while (!terminate && (int)out_ids.size() < n_steps) {
    const int budget = n_steps - (int)out_ids.size();
    const int L = adaptive ? _dflash_tuner.pick(std::min(budget, Lmax))
                           : std::min(Lb, budget);
    const int pos = _ctx->seq_len_of(cid);    // the anchor's KV slot
    const auto t0 = clk::now();
    // One buffer: what the last round owes, then the draft.
    {
      metal_compute::CommandStream& ds = spec_stream_();
      {
        ComputeEncoder enc = ds.begin_compute();
        encode_owed(enc);
        if (L >= 2) {
          _dflash->fill_block_ids(bp, anchor, L);
          _dflash->encode_draft(enc, *hooks, blk, L, pos, drafts, 1);
        }
      }
      if (!ds.commit().wait_ok()) { failed = true; break; }
    }
    vids.assign(1, anchor);
    for (int i = 1; i < L; ++i) { vids.push_back(dp[i]); }
    const auto t1 = clk::now();
    // The verify, tapping the drafter's layers at every row.
    const int g0 = (int)out_ids.size();     // the anchor's index in the run
    const int rp = ctl.rope_first < 0 ? -1 : ctl.rope_first + (pos - main0);
    // The verifier's pick at every row, on the GPU as a decode step picks
    // -- argmax, or a sample seeded by the token's index in the run,
    // against the seen-set a step would hold there -- encoded at the
    // verify's end.
    if (penal) {
      auto* sr = static_cast<std::uint8_t*>(seen_rows.contents());
      for (int i = 0; i < L; ++i) {
        std::uint8_t* rw = sr + (std::size_t)i * V;
        std::memcpy(rw, seen_base.data(), V);
        for (int q = 0; q <= i; ++q) {
          const std::int32_t t = vids[(std::size_t)q];
          if (t >= 0 && (std::size_t)t < V) { rw[(std::size_t)t] = 1; }
        }
      }
    }
    HcVerify hv;
    hv.ring = true;
    HcRun vr;
    vr.verify = &hv;
    vr.all_rows = true;
    vr.tap_layers = &tl;
    vr.taps = &taps;
    vr.tail = [&](ComputeEncoder& enc) {
      for (int i = 0; i < L; ++i) {
        const std::size_t lo = (std::size_t)i * V * 2;
        if (!sampling) {
          encode_argmax_(enc, _hc_logits, lo, picks, (std::size_t)i * 4,
                         c.vocab);
        } else {
          const std::uint32_t seed = (std::uint32_t)(
              sp.seed + 0x9e3779b9ull * (std::uint64_t)(g0 + i + 1));
          encode_sample_core_(enc, _hc_logits, lo, picks,
                              (std::size_t)i * 4, ws, 0, seen_rows,
                              (std::size_t)i * V, sp, seed, c.vocab);
        }
      }
    };
    const bool vok = !forward_hc_(cid, vids, nullptr, nullptr, nullptr, rp,
                                  nullptr, &vr).empty();
    if (!vok) {
      const int extra = _ctx->seq_len_of(cid) - pos;
      if (extra > 0) { _ctx->kv_rollback(cid, extra); }
      failed = true;
      break;
    }
    preds.assign(static_cast<const std::int32_t*>(picks.contents()),
                 static_cast<const std::int32_t*>(picks.contents()) + L);
    const double round_ms =
        std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    st.gpu_ms += round_ms;
    ++st.rounds;
    // The longest prefix the verifier agrees with; the anchor always is.
    int j = 1;
    while (j < L && vids[(std::size_t)j] == preds[(std::size_t)j - 1]) {
      ++j;
    }
    if (adaptive) { _dflash_tuner.record(L, round_ms, j - 1); }
    int keep = j;
    // A stop token ends the decode and is not kept (as mtp_decode).
    if (ctl.is_stop) {
      for (int i = 0; i < keep; ++i) {
        if (ctl.is_stop(vids[(std::size_t)i])) {
          keep = i;
          terminate = true;
          if (ctl.hit_stop) { *ctl.hit_stop = true; }
          break;
        }
      }
    }
    keep = std::min(keep, n_steps - (int)out_ids.size());
    for (int i = 0; i < keep; ++i) {
      out_ids.push_back(vids[(std::size_t)i]);
      if (penal) {
        const std::int32_t t = vids[(std::size_t)i];
        if (t >= 0 && (std::size_t)t < V) { seen_base[(std::size_t)t] = 1; }
      }
    }
    st.drafted += L - 1;
    if (keep > 1) { st.accepted += keep - 1; }
    // The rejected rows out of the KV; the GDN and PLE state the kept
    // rows' (slot 1 whole, else replayed into it -- owed to the next
    // buffer), and the kept rows' taps to the drafter (owed likewise).
    if (keep < L) { _ctx->kv_rollback(cid, L - keep); }
    if (keep == L) {
      _ctx->gdn_ring_advance(cid);
    } else if (keep > 0) {
      owe_keep = keep;
      hv_owed = std::move(hv);
      vids_owed = vids;
    }
    if (keep > 0) {
      owe_rows = keep;
      owe_pos = pos;
      owe_L = L;
    }
    if (kProf) {
      t_draft += std::chrono::duration<double, std::milli>(t1 - t0).count();
      t_verify += std::chrono::duration<double, std::milli>(
          clk::now() - t1).count();
    }
    if (ctl.on_round && keep > 0) {
      const std::span<const std::int32_t> fresh(
          out_ids.data() + (out_ids.size() - (std::size_t)keep),
          (std::size_t)keep);
      if (!ctl.on_round(fresh)) { terminate = true; }
    }
    if (keep == 0) { terminate = true; }
    if (!terminate) { anchor = preds[(std::size_t)keep - 1]; }
  }
  // Settle what the last round owes: the context's state the kept
  // tokens', and the drafter having seen them.
  if (!failed && (owe_keep > 0 || owe_rows > 0)) {
    metal_compute::CommandStream& fs = spec_stream_();
    {
      ComputeEncoder enc = fs.begin_compute();
      encode_owed(enc);
    }
    failed = !fs.commit().wait_ok();
  }
  _ctx->gdn_ring_end(cid);
  if (stats) { *stats = st; }
  if (ctl.drafted) { *ctl.drafted = st.drafted; }
  if (kProf && st.rounds > 0) {
    if (const SessionContextIntf* s = _mc->session()) {
      s->info(fmt("[dflash] {} on Qwen3.8-Flash-Next, block {}: {} tokens in "
                  "{} rounds ({:.2f} a round), {} of {} drafts kept; per "
                  "round draft {:.1f} ms, verify {:.1f} ms",
                  dc.kind(), adaptive ? std::string("auto")
                                      : std::to_string(Lb),
                  out_ids.size(), st.rounds,
                  (double)out_ids.size() / (double)st.rounds, st.accepted,
                  st.drafted, t_draft / (double)st.rounds,
                  t_verify / (double)st.rounds));
    }
  }
  return !failed || !out_ids.empty();
}

}  // namespace vpipe::genai
