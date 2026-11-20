#include "generative-models/relayvsr/metal-relayvsr-wan.h"

#include "common/flex-bag.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/relayvsr/relayvsr-ops.h"
#include "generative-models/shared/ane-ffn.h"
#include "generative-models/shared/fp8-layout.h"
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
using relayvsr::from_bf16;
using relayvsr::to_bf16;

namespace {

constexpr const char* kKey = "relayvsr-wan/";

float
bf(float v)
{
  return from_bf16(to_bf16(v));
}

// y = bf16(W x + b) on the host, bf16 operands, f32 accumulation -- how a
// bf16 Linear lands on the reference's GPU. W is read where the file's
// bytes were read to, never widened into a host copy (the projection is
// 226 MB as f32, and macOS keeps a freed large block in the footprint).
std::vector<float>
linear_host(const relayvsr::RawTensor& w, const std::vector<float>& b,
            const std::vector<float>& x, int N, int K)
{
  std::vector<float> y((std::size_t)N);
  for (int n = 0; n < N; ++n) {
    const std::size_t row = (std::size_t)n * K;
    double acc = 0.0;
    for (int k = 0; k < K; ++k) {
      acc += (double)w.at(row + (std::size_t)k) * (double)x[k];
    }
    y[(std::size_t)n] = bf((float)acc + (b.empty() ? 0.0f : b[(size_t)n]));
  }
  return y;
}

float
silu_bf(float v)
{
  return bf(v / (1.0f + std::exp(-v)));
}

// ComfyUI'S NAMES (Comfy-Org/RelayVSR's repack) for the release's diffusers
// ones -- a pure renaming, measured against the release: no row or column
// moves (the int8 projections dequantized sit at cosine 0.99996 to the
// bf16 file's, every block).
std::string
comfy_name(const std::string& d)
{
  static const std::pair<const char*, const char*> kTop[] = {
      {"lq_proj.out_proj.", "lq_proj_in.linear."},
      {"lq_proj.", "lq_proj_in."},
      {"proj_out.", "head.head."},
      {"scale_shift_table", "head.modulation"},
      {"condition_embedder.time_embedder.linear_1.", "time_embedding.0."},
      {"condition_embedder.time_embedder.linear_2.", "time_embedding.2."},
      {"condition_embedder.time_proj.", "time_projection.1."},
      {"condition_embedder.text_embedder.linear_1.", "text_embedding.0."},
      {"condition_embedder.text_embedder.linear_2.", "text_embedding.2."},
      {"prompt_embeds", "positive_conditioning"},
  };
  static const std::pair<const char*, const char*> kBlock[] = {
      {"attn1.to_q.", "self_attn.q."},     {"attn1.to_k.", "self_attn.k."},
      {"attn1.to_v.", "self_attn.v."},     {"attn1.to_out.0.", "self_attn.o."},
      {"attn1.norm_q.", "self_attn.norm_q."},
      {"attn1.norm_k.", "self_attn.norm_k."},
      {"attn2.to_q.", "cross_attn.q."},    {"attn2.to_k.", "cross_attn.k."},
      {"attn2.to_v.", "cross_attn.v."},    {"attn2.to_out.0.", "cross_attn.o."},
      {"attn2.norm_q.", "cross_attn.norm_q."},
      {"attn2.norm_k.", "cross_attn.norm_k."},
      {"ffn.net.0.proj.", "ffn.0."},       {"ffn.net.2.", "ffn.2."},
      {"norm2.", "norm3."},                {"scale_shift_table", "modulation"},
  };
  auto starts = [](const std::string& a, const char* b) {
    return a.compare(0, std::strlen(b), b) == 0;
  };
  if (starts(d, "blocks.")) {
    const std::size_t dot = d.find('.', 7);
    if (dot == std::string::npos) { return d; }
    const std::string head = d.substr(0, dot + 1), rest = d.substr(dot + 1);
    for (const auto& [a, b] : kBlock) {
      if (starts(rest, a)) { return head + b + rest.substr(std::strlen(a)); }
    }
    return d;
  }
  for (const auto& [a, b] : kTop) {
    if (starts(d, a)) { return b + d.substr(std::strlen(a)); }
  }
  return d;
}

// Whether `w` spells the model in ComfyUI's names.
bool
comfy_spelled(const MetalLlamaWeights& w)
{
  return w.info("blocks.0.self_attn.q.weight") != nullptr;
}

}  // namespace

struct MetalRelayVsrWan::Block {
  relayvsr::Proj q, k, v, o, xq, xo, f1, f2;
  SharedBuffer nq, nk, xnq, n2w, n2b;
  // Folded: the modulation, f32 [6, hidden] (shift, scale, gate, then the
  // feed-forward's three), and the cross-attention's K and V over the
  // fixed context, [text_tokens, hidden] each.
  SharedBuffer mod, ck, cv;
};

struct MetalRelayVsrWan::Scratch {
  int ph = 0, pw = 0, n = 0;
  SharedBuffer lqr, lq1, lq2, prow, x, y, q, a, o, ff, kr, head, noise;
  std::vector<SharedBuffer> ring_k, ring_v;   // per block
};

MetalRelayVsrWan::~MetalRelayVsrWan()
{
  if (_ane) { _ane->join(); }
}

bool
MetalRelayVsrWan::uses_matrix_cores() const noexcept
{
  return _ops && _ops->matrix_cores();
}

bool
MetalRelayVsrWan::window_on_matrix_cores() const noexcept
{
  return _ops && _ops->nax_attention();
}

bool
MetalRelayVsrWan::cross_on_matrix_cores() const noexcept
{
  return _ops && _ops->cross_on_matrix_cores();
}

bool
MetalRelayVsrWan::config_from_checkpoint(const std::string& wan_file,
                                         Config* out, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  auto w = MetalLlamaWeights::open(wan_file);
  if (!w) { return fail(fmt("'{}' is not a safetensors file", wan_file)()); }
  const auto& md = w->metadata();
  auto it = md.find("format");
  const bool x2 = it != md.end() && it->second == "relayvsr-wan-x2-v1";
  if (it == md.end() || (it->second != "refvsr-wan-v1" && !x2)) {
    return fail(fmt("'{}' is not a RelayVSR keyframe checkpoint (format "
                    "refvsr-wan-v1, or relayvsr-wan-x2-v1)", wan_file)());
  }
  auto ct = md.find("config");
  if (ct == md.end()) { return fail("refvsr-wan: no config in the file"); }
  const FlexData c = FlexData::from_json(ct->second);
  Config cfg;
  // The release's 4x metadata predates the scale; the x2 file states it.
  cfg.upscale = (int)bag::integer(c, "upscale_factor", x2 ? 2 : 4);
  // ComfyUI's int8 projections (its repack), read as stored.
  {
    fp8::Layout lay;
    std::string le;
    if (!fp8::scan(*w, &lay, &le, /*int8=*/true)) {
      return fail(fmt("refvsr-wan: '{}' -- {}", wan_file, le)());
    }
    for (const auto& [nm, q] : lay.weights) {
      if (q.format != fp8::Format::kI8) {
        return fail(fmt("refvsr-wan: '{}' stores '{}' in an 8-bit format "
                        "this port does not run (only ComfyUI's int8)",
                        wan_file, nm)());
      }
    }
    cfg.int8 = lay.any();
  }
  cfg.heads = (int)bag::integer(c, "num_attention_heads", 24);
  cfg.head_dim = (int)bag::integer(c, "attention_head_dim", 128);
  cfg.hidden = cfg.heads * cfg.head_dim;
  cfg.in_channels = (int)bag::integer(c, "in_channels", 48);
  cfg.text_dim = (int)bag::integer(c, "text_dim", 4096);
  cfg.freq_dim = (int)bag::integer(c, "freq_dim", 256);
  cfg.ffn = (int)bag::integer(c, "ffn_dim", 14336);
  cfg.layers = (int)bag::integer(c, "num_layers", 30);
  cfg.eps = (float)bag::real(c, "eps", 1e-6);
  cfg.lq_hidden1 = (int)bag::integer(c, "lq_proj_hidden_dim1", 768);
  cfg.lq_hidden2 = (int)bag::integer(c, "lq_proj_hidden_dim2", 3072);
  cfg.context_frames = (int)bag::integer(c, "temporal_context_frames", 4);
  const FlexData win = bag::value(c, "spatial_window_size");
  const FlexData rope = bag::value(c, "temporal_rope");
  cfg.rope_frames = (int)bag::integer(rope, "training_frames", 8);
  // What this port runs, said where the file says otherwise.
  std::string why;
  const FlexData patch = bag::value(c, "patch_size");
  const char* c1 = comfy_spelled(*w) ? "lq_proj_in.conv1.weight"
                                     : "lq_proj.conv1.weight";
  const auto* c1i = w->info(c1);
  if (cfg.upscale != (x2 ? 2 : 4)) {
    why = fmt("upscale_factor {} in a {} file", cfg.upscale, it->second)();
  } else if (c1i == nullptr || c1i->shape.size() != 4 ||
             c1i->shape[1] != cfg.lq_channels()) {
    why = fmt("its input projection does not read a {}x frame "
              "({} channels)", cfg.upscale, cfg.lq_channels())();
  } else if (!patch.is_array() || patch.as_array().size() != 3 ||
      patch.as_array()[0].as_int() != 1 || patch.as_array()[1].as_int() != 2 ||
      patch.as_array()[2].as_int() != 2) {
    why = "patch_size is not [1, 2, 2]";
  } else if (bag::integer(c, "out_channels", 48) != cfg.in_channels) {
    why = "out_channels differs from in_channels";
  } else if (cfg.head_dim != 128) {
    why = "attention_head_dim is not 128";
  } else if (!bag::flag(c, "cross_attn_norm", true)) {
    why = "cross_attn_norm is off";
  } else if (bag::text(c, "qk_norm", "rms_norm_across_heads") !=
             "rms_norm_across_heads") {
    why = "qk_norm is not rms_norm_across_heads";
  } else if (!bag::flag(c, "use_block_causal_attention", false) ||
             bag::integer(c, "num_frames_per_block", 1) != 1 ||
             bag::integer(c, "sink_size", 0) != 0) {
    why = "it is not one-frame block-causal streaming without sinks";
  } else if (bag::text(c, "spatial_attention_mode", "local") != "local" ||
             !win.is_array() || win.as_array().size() != 2) {
    why = "its spatial attention is not a local window";
  } else if (bag::text(rope, "policy", "fixed_end") != "fixed_end") {
    why = "temporal_rope.policy is not fixed_end";
  } else if (cfg.context_frames < 1 || cfg.context_frames > 4 ||
             cfg.context_frames > cfg.rope_frames) {
    why = fmt("temporal_context_frames {} (1..4, at most the rope's {})",
              cfg.context_frames, cfg.rope_frames)();
  }
  if (!why.empty()) {
    return fail(fmt("refvsr-wan: '{}' -- {}; this port runs the release's "
                    "configuration only", wan_file, why)());
  }
  cfg.window_h = (int)win.as_array()[0].as_int();
  cfg.window_w = (int)win.as_array()[1].as_int();
  if (cfg.window_h <= 0 || cfg.window_w <= 0) {
    return fail("refvsr-wan: a spatial window of no size");
  }
  *out = cfg;
  return true;
}

std::uint64_t
MetalRelayVsrWan::ring_bytes(const Config& cfg, int ph, int pw)
{
  const int ts = cfg.token_stride();
  const std::uint64_t n = (std::uint64_t)(ph / ts) * (pw / ts);
  return (std::uint64_t)cfg.layers * 2u * (std::uint64_t)cfg.context_frames *
         n * (std::uint64_t)cfg.hidden * 2u;
}

std::uint64_t
MetalRelayVsrWan::scratch_bytes(const Config& cfg, int ph, int pw)
{
  const int ts = cfg.token_stride();
  const std::uint64_t n = (std::uint64_t)(ph / ts) * (pw / ts);
  const std::uint64_t D = (std::uint64_t)cfg.hidden;
  const std::uint64_t n4 = 4 * n;
  std::uint64_t b = 0;
  b += n4 * (std::uint64_t)cfg.lq_channels() * 2u;          // lqr
  b += n4 * (std::uint64_t)cfg.lq_hidden1 * 2u;             // lq1
  b += n * (std::uint64_t)cfg.lq_hidden2 * 2u;     // lq2
  b += n * 4u * (std::uint64_t)cfg.in_channels * 2u;   // prow
  b += 5u * n * D * 2u;                            // x y q a o
  b += n * (std::uint64_t)cfg.ffn * 2u;            // ff
  b += (std::uint64_t)cfg.context_frames * n * D * 2u;   // kr
  b += n * 4u * (std::uint64_t)cfg.in_channels * 2u;     // head
  b += n4 * (std::uint64_t)cfg.in_channels * 4u * 2u;   // noise, latent
  // The int8 context's over the blocks' projections: q k v o and the
  // cross-attention's q o [D, D], the feed-forward's up [F, D] and down
  // [D, F] (on the accurate tier the down's 14336 deep, so its tuner
  // takes the split-K budget).
  const int Di = cfg.hidden, Fi = cfg.ffn;
  b += relayvsr::Ops::i8_scratch_bytes(cfg.i8_tier, cfg.autotune, (int)n,
                                       {{Di, Di}, {Fi, Di}, {Di, Fi}});
  // SageAttention's int8 operands, over every slot of the ring, and the
  // cross-attention's over the context.
  b += relayvsr::Ops::window_int8_bytes(cfg.context_frames, cfg.heads,
                                        cfg.heads, (int)n, cfg.head_dim,
                                        cfg.sage_mode());
  b += relayvsr::Ops::cross_int8_bytes(cfg.heads, (int)n, cfg.text_tokens,
                                       cfg.sage_mode());
  // ComfyUI's int8 widened to bf16 before each GEMM no tier holds: one
  // copy of the widest (the feed-forward's).
  b += relayvsr::Ops::widen_bytes(cfg.i8_tier, cfg.int8, {Di, Fi, Di},
                                  {Di, Di, Fi});
  return b;
}

std::uint64_t
MetalRelayVsrWan::weight_bytes(const Config& cfg)
{
  const std::uint64_t D = (std::uint64_t)cfg.hidden;
  const std::uint64_t F = (std::uint64_t)cfg.ffn;
  // A block: q k v o, cross q o (k v folded away), the feed-forward -- at
  // the tier's layout -- their biases, the norms; plus its folded
  // modulation (f32) and context K / V.
  const int t = cfg.i8_tier;
  const bool q = cfg.int8;
  const std::uint64_t blk =
      6u * relayvsr::proj_bytes((int)D, (int)D, t, q) +
      relayvsr::proj_bytes((int)F, (int)D, t, q) +
      relayvsr::proj_bytes((int)D, (int)F, t, q) +
      (6u * D + F + D + 5u * D) * 2u + 6u * D * 4u +
      2u * (std::uint64_t)cfg.text_tokens * D * 2u;
  const std::uint64_t h1 = (std::uint64_t)cfg.lq_hidden1;
  const std::uint64_t h2 = (std::uint64_t)cfg.lq_hidden2;
  const std::uint64_t pv = 4u * (std::uint64_t)cfg.in_channels;
  const std::uint64_t lc = (std::uint64_t)cfg.lq_channels();
  const std::uint64_t top =
      (lc * h1 + 2u * h1 + 9u * h1 * h2 + 2u * h2 + h2 * D + D +
       pv * D + D + D * pv + pv) * 2u + 2u * D * 4u +
      2u * 1024u * (std::uint64_t)cfg.head_dim * 4u;
  return (std::uint64_t)cfg.layers * blk + top;
}

std::uint64_t
MetalRelayVsrWan::held_bytes() const
{
  std::uint64_t b = weight_bytes(_cfg);
  for (const Block& k : _blocks) {
    for (const relayvsr::Proj* p :
         {&k.q, &k.k, &k.v, &k.o, &k.xq, &k.xo, &k.f1, &k.f2}) {
      b += p->weight_bytes();
      b -= relayvsr::proj_bytes(p->n, p->k, _cfg.i8_tier, _cfg.int8);
    }
  }
  return b;
}

const relayvsr::Ops&
MetalRelayVsrWan::ops() const noexcept
{
  return *_ops;
}

bool
MetalRelayVsrWan::ane_stages(const Config& cfg)
{
  // The up projection is hidden deep, the down ffn deep.
  return relayvsr::held_group(cfg.hidden, cfg.i8_tier) == 0 &&
         relayvsr::held_group(cfg.ffn, cfg.i8_tier) == 0;
}

std::uint64_t
MetalRelayVsrWan::ane_runtime_bytes(const Config& cfg)
{
  // The chunk a large forward takes (or VPIPE_RELAYVSR_ANE_CHUNK's).
  return AneFeedForward::gelu_runtime_bytes(
      cfg.hidden, cfg.ffn, relayvsr::ane_chunk(1 << 30));
}

std::uint64_t
MetalRelayVsrWan::ane_held_bytes() const
{
  return _ane ? AneFeedForward::gelu_runtime_bytes(_cfg.hidden, _cfg.ffn,
                                                   _ane->chunk())
              : 0;
}

std::unique_ptr<MetalRelayVsrWan>
MetalRelayVsrWan::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                       const Config& cfg, std::string* err)
{
  auto fail = [&](const std::string& m) -> std::unique_ptr<MetalRelayVsrWan> {
    if (err != nullptr) { *err = m; }
    return nullptr;
  };
  if (!ws || mc == nullptr) { return fail("no weights or backend"); }
  auto m = std::unique_ptr<MetalRelayVsrWan>(new MetalRelayVsrWan());
  m->_cfg = cfg;
  m->_mc = mc;
  m->_ws = std::move(ws);
  m->_ops = std::make_unique<relayvsr::Ops>();
  if (!m->_ops->load(mc, /*bf16=*/true, err)) { return nullptr; }
  // The int8 tier first: a tier that holds the layout converts each block
  // projection as it loads. What it settles on (the context declines
  // without matrix cores) is the tier recorded.
  (void)m->_ops->set_i8(cfg.i8_tier, cfg.autotune);
  m->_cfg.i8_tier = m->_ops->i8_tier();
  // The self-attention's int8 prologue at the ring's every slot.
  m->_ops->set_window_planes(cfg.context_frames);
  WeightSet& w = *m->_ws;
  // The file's spelling: the release's names, or ComfyUI's -- and its
  // int8 projections, which the file (not the caller) says it has.
  const bool comfy = comfy_spelled(w.src());
  m->_comfy = comfy;
  auto nm_ = [comfy](const std::string& d) {
    return comfy ? comfy_name(d) : d;
  };
  {
    std::string le;
    if (!fp8::scan(w.src(), &m->_f8, &le, /*int8=*/true)) {
      return fail("refvsr-wan: " + le);
    }
    m->_cfg.int8 = m->_f8.any();
  }
  auto t = [&](const std::string& nm) {
    return relayvsr::tensor_elt(w, mc, nm_(nm), true, kKey);
  };
  // The 3x3 conv's weight, tap-major for the gather conv: [Cout, 9 * Cin].
  {
    const std::string nm = nm_("lq_proj.conv2.weight");
    m->_lq_c2w = w.derived(std::string(kKey) + "tap-major/" + nm, [&]() {
      relayvsr::RawTensor r;
      if (!relayvsr::read_raw(w, mc, nm, &r)) { return SharedBuffer{}; }
      return relayvsr::tap_major_elt(mc, r, true);
    });
  }
  m->_lq_c1w = t("lq_proj.conv1.weight");
  m->_lq_c1b = t("lq_proj.conv1.bias");
  m->_lq_n1 = t("lq_proj.norm1.weight");
  m->_lq_c2b = t("lq_proj.conv2.bias");
  m->_lq_n2 = t("lq_proj.norm2.weight");
  m->_lq_ow = t("lq_proj.out_proj.weight");
  m->_lq_ob = t("lq_proj.out_proj.bias");
  m->_patch_w = t("patch_embedding.weight");
  m->_patch_b = t("patch_embedding.bias");
  m->_head_w = t("proj_out.weight");
  m->_head_b = t("proj_out.bias");
  for (const SharedBuffer* b :
       {&m->_lq_c1w, &m->_lq_c1b, &m->_lq_n1, &m->_lq_c2w, &m->_lq_c2b,
        &m->_lq_n2, &m->_lq_ow, &m->_lq_ob, &m->_patch_w, &m->_patch_b,
        &m->_head_w, &m->_head_b}) {
    if (b->empty()) {
      return fail("refvsr-wan: a top-level tensor is missing");
    }
  }
  m->_blocks.resize((std::size_t)cfg.layers);
  relayvsr::Ops& op = *m->_ops;
  const int D = cfg.hidden, F = cfg.ffn;
  for (int i = 0; i < cfg.layers; ++i) {
    Block& b = m->_blocks[(std::size_t)i];
    const std::string p = fmt("blocks.{}.", i)();
    std::string pe;
    auto proj = [&](const std::string& nm, int N, int K) {
      const std::string wn = nm_(p + nm + ".weight");
      return op.load_proj(w, wn, nm_(p + nm + ".bias"), N, K, kKey, true,
                          m->_f8.find(wn), true, &pe);
    };
    b.q = proj("attn1.to_q", D, D);
    b.k = proj("attn1.to_k", D, D);
    b.v = proj("attn1.to_v", D, D);
    b.o = proj("attn1.to_out.0", D, D);
    b.xq = proj("attn2.to_q", D, D);
    b.xo = proj("attn2.to_out.0", D, D);
    b.f1 = proj("ffn.net.0.proj", F, D);
    b.f2 = proj("ffn.net.2", D, F);
    b.nq = t(p + "attn1.norm_q.weight"); b.nk = t(p + "attn1.norm_k.weight");
    b.xnq = t(p + "attn2.norm_q.weight");
    b.n2w = t(p + "norm2.weight"); b.n2b = t(p + "norm2.bias");
    bool ok = true;
    for (const relayvsr::Proj* pr :
         {&b.q, &b.k, &b.v, &b.o, &b.xq, &b.xo, &b.f1, &b.f2}) {
      ok = ok && pr->valid();
    }
    for (const SharedBuffer* s : {&b.nq, &b.nk, &b.xnq, &b.n2w, &b.n2b}) {
      ok = ok && !s->empty();
    }
    if (!ok) {
      return fail(pe.empty() ? fmt("refvsr-wan: block {} is incomplete", i)()
                             : pe);
    }
  }
  if (!m->fold_(err)) { return nullptr; }
  return m;
}

// THE CONSTANTS. One timestep and one context, so everything that hangs
// off them is computed once: the timestep embedding and its projection on
// the host (a few dozen million MACs), the context through the text
// embedder and every block's cross K and V on the GPU.
bool
MetalRelayVsrWan::fold_(std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  WeightSet& w = *_ws;
  const int D = _cfg.hidden, FD = _cfg.freq_dim, T = _cfg.text_tokens;
  const bool comfy = _comfy;
  auto nm_ = [comfy](const std::string& d) {
    return comfy ? comfy_name(d) : d;
  };
  auto rd = [&](const std::string& nm, std::vector<float>* v) {
    return relayvsr::read_f32(w, _mc, nm_(nm), v);
  };
  // ---- the timestep: [cos | sin] of t over FD/2 frequencies, bf16, then
  // linear_1 -> silu -> linear_2 (temb), silu -> time_proj.
  std::vector<float> emb((std::size_t)FD);
  {
    const int half = FD / 2;
    for (int i = 0; i < half; ++i) {
      const float ex = -std::log(10000.0f) * (float)i / (float)half;
      const float a = _cfg.timestep * std::exp(ex);
      emb[(std::size_t)i] = bf(std::cos(a));
      emb[(std::size_t)(i + half)] = bf(std::sin(a));
    }
  }
  relayvsr::RawTensor w1, w2, wp;
  std::vector<float> b1, b2, bp;
  auto raw = [&](const std::string& nm, int N, int K,
                 relayvsr::RawTensor* t) {
    return relayvsr::read_raw(w, _mc, nm_(nm), t) &&
           t->n == (std::size_t)N * (std::size_t)K;
  };
  if (!raw("condition_embedder.time_embedder.linear_1.weight", D, FD, &w1) ||
      !rd("condition_embedder.time_embedder.linear_1.bias", &b1) ||
      !raw("condition_embedder.time_embedder.linear_2.weight", D, D, &w2) ||
      !rd("condition_embedder.time_embedder.linear_2.bias", &b2) ||
      !raw("condition_embedder.time_proj.weight", 6 * D, D, &wp) ||
      !rd("condition_embedder.time_proj.bias", &bp)) {
    return fail("refvsr-wan: the timestep embedder is missing");
  }
  std::vector<float> h = linear_host(w1, b1, emb, D, FD);
  for (float& v : h) { v = silu_bf(v); }
  const std::vector<float> temb = linear_host(w2, b2, h, D, D);
  std::vector<float> st(temb);
  for (float& v : st) { v = silu_bf(v); }
  const std::vector<float> tp = linear_host(wp, bp, st, 6 * D, D);
  for (int i = 0; i < _cfg.layers; ++i) {
    std::vector<float> sst;
    if (!rd(fmt("blocks.{}.scale_shift_table", i)(), &sst) ||
        sst.size() != (std::size_t)6 * D) {
      return fail(fmt("refvsr-wan: block {} has no modulation table", i)());
    }
    // In f32, as the reference keeps it: table.float() + temb.float().
    std::vector<float> mod((std::size_t)6 * D);
    for (std::size_t j = 0; j < mod.size(); ++j) { mod[j] = sst[j] + tp[j]; }
    // Kept in the table's order: shift, scale, gate, then the feed-
    // forward's three (the kernels take each by offset).
    _blocks[(std::size_t)i].mod = relayvsr::upload_f32(_mc, mod);
  }
  {
    // The head's: (table + temb) in bf16, and 1 + scale in bf16 too.
    std::vector<float> sst;
    if (!rd("scale_shift_table", &sst) || sst.size() != (std::size_t)2 * D) {
      return fail("refvsr-wan: the head has no modulation table");
    }
    std::vector<float> sh((std::size_t)D), sc((std::size_t)D);
    for (int j = 0; j < D; ++j) {
      sh[(std::size_t)j] = bf(sst[(std::size_t)j] + temb[(std::size_t)j]);
      const float s = bf(sst[(std::size_t)(D + j)] + temb[(std::size_t)j]);
      sc[(std::size_t)j] = bf(1.0f + s) - 1.0f;
    }
    _head_shift = relayvsr::upload_f32(_mc, sh);
    _head_scale = relayvsr::upload_f32(_mc, sc);
  }
  // ---- RoPE: diffusers' Wan tables, angles in f64, cos/sin as f32.
  {
    const int HD = _cfg.head_dim;
    const int hd = 2 * (HD / 6), td = HD - 2 * hd;
    std::vector<float> c((std::size_t)1024 * HD), s((std::size_t)1024 * HD);
    for (int pos = 0; pos < 1024; ++pos) {
      int col = 0;
      for (const int d : {td, hd, hd}) {
        for (int i = 0; i < d / 2; ++i) {
          const double inv = 1.0 / std::pow(10000.0, (double)(2 * i) / d);
          const double a = (double)pos * inv;
          for (int r = 0; r < 2; ++r) {
            c[(std::size_t)pos * HD + col] = (float)std::cos(a);
            s[(std::size_t)pos * HD + col] = (float)std::sin(a);
            ++col;
          }
        }
      }
    }
    _rope_cos = relayvsr::upload_f32(_mc, c);
    _rope_sin = relayvsr::upload_f32(_mc, s);
  }
  // ---- the context: prompt_embeds -> linear_1 -> gelu(tanh) -> linear_2,
  // then each block's K = norm_k(to_k ctx) and V = to_v ctx. The cross K /
  // V projections are read UNCACHED: consumed here, never held.
  relayvsr::Ops& op = *_ops;
  // The release's [1, 512, 4096]; ComfyUI's [512, 4096].
  const std::string pn = nm_("prompt_embeds");
  SharedBuffer pe = w.read(pn, _mc, WeightSet::Residency::Copied);
  const auto* pinfo = w.src().info(pn);
  if (pe.empty() || pinfo == nullptr || pinfo->dtype != "BF16" ||
      pinfo->shape.size() < 2 || pinfo->shape[pinfo->shape.size() - 2] != T ||
      pinfo->shape.back() != _cfg.text_dim) {
    return fail("refvsr-wan: prompt_embeds is not bf16 [1, 512, 4096]");
  }
  auto rdb = [&](const std::string& nm) {
    return w.read(nm_(nm), _mc, WeightSet::Residency::Copied);
  };
  SharedBuffer l1w = rdb("condition_embedder.text_embedder.linear_1.weight");
  SharedBuffer l1b = rdb("condition_embedder.text_embedder.linear_1.bias");
  SharedBuffer l2w = rdb("condition_embedder.text_embedder.linear_2.weight");
  SharedBuffer l2b = rdb("condition_embedder.text_embedder.linear_2.bias");
  if (l1w.empty() || l1b.empty() || l2w.empty() || l2b.empty()) {
    return fail("refvsr-wan: the text embedder is missing");
  }
  SharedBuffer c1 = _mc->make_shared_buffer((std::size_t)T * D * 2);
  SharedBuffer ctx = _mc->make_shared_buffer((std::size_t)T * D * 2);
  if (c1.empty() || ctx.empty()) { return fail("context allocation failed"); }
  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    op.gemm(enc, pe, 0, l1w, &l1b, c1, 0, T, D, _cfg.text_dim);
    enc.set_function(op.gelu);
    enc.set_buffer(0, c1); enc.set_buffer(1, c1);
    enc.set_constant(2, T * D);
    enc.dispatch({(unsigned)(T * D), 1, 1}, {256, 1, 1});
    op.gemm(enc, c1, 0, l2w, &l2b, ctx, 0, T, D, D);
    enc.end();
    std::string ge;
    if (!stream.commit().wait_ok(&ge)) { return fail("context: " + ge); }
  }
  for (int i = 0; i < _cfg.layers; ++i) {
    Block& b = _blocks[(std::size_t)i];
    const std::string p = fmt("blocks.{}.attn2.", i)();
    // Read UNCACHED and consumed (a ComfyUI int8 weight widened to bf16 on
    // the way, as the forward widens one): the dense GEMM either way.
    std::string pe2;
    auto cproj = [&](const char* nm) {
      const std::string wn = nm_(p + nm + ".weight");
      return op.load_proj(w, wn, nm_(p + nm + ".bias"), D, D, kKey,
                          /*hold=*/false, _f8.find(wn), /*cache=*/false,
                          &pe2);
    };
    const relayvsr::Proj kp = cproj("to_k");
    const relayvsr::Proj vp = cproj("to_v");
    SharedBuffer nk = rdb(p + "norm_k.weight");
    if (!kp.valid() || !vp.valid() || nk.empty()) {
      return fail(pe2.empty()
                      ? fmt("refvsr-wan: block {}'s cross K / V is missing",
                            i)()
                      : pe2);
    }
    b.ck = _mc->make_shared_buffer((std::size_t)T * D * 2);
    b.cv = _mc->make_shared_buffer((std::size_t)T * D * 2);
    if (b.ck.empty() || b.cv.empty()) { return fail("context K/V allocation"); }
    ComputeEncoder enc = stream.begin_compute();
    const SharedBuffer* kw = op.dense_weight(enc, kp);
    if (kw == nullptr) { return fail("refvsr-wan: cross K widening"); }
    op.gemm(enc, ctx, 0, *kw, &kp.b, b.ck, 0, T, D, D);
    enc.set_function(op.rms);
    enc.set_buffer(0, b.ck); enc.set_buffer(1, nk); enc.set_buffer(2, b.ck);
    enc.set_constant(3, D); enc.set_constant(4, _cfg.eps);
    enc.dispatch({256, (unsigned)T, 1}, {256, 1, 1});
    const SharedBuffer* vw = op.dense_weight(enc, vp);
    if (vw == nullptr) { return fail("refvsr-wan: cross V widening"); }
    op.gemm(enc, ctx, 0, *vw, &vp.b, b.cv, 0, T, D, D);
    enc.end();
    std::string ge;
    if (!stream.commit().wait_ok(&ge)) { return fail("context K/V: " + ge); }
  }
  // The fold's widenings are done. Where a forward widens too (ComfyUI's
  // int8 off the held tiers) the scratch, made at the widest such
  // projection, is the forward's; where none does, it goes -- the plan
  // books it only where a forward widens (MEASURED, 2x token: 12 MB over
  // the booking with the fold's copy kept).
  if (!op.widens()) { op.drop_widened(); }
  return true;
}

void
MetalRelayVsrWan::reset()
{
  _frame = 0;
}

bool
MetalRelayVsrWan::ensure_scratch_(int ph, int pw, std::string* err)
{
  if (_sc && _sc->ph == ph && _sc->pw == pw) { return true; }
  _sc.reset();
  auto sc = std::make_unique<Scratch>();
  sc->ph = ph;
  sc->pw = pw;
  const int ts = _cfg.token_stride();
  const std::size_t n = (std::size_t)(ph / ts) * (pw / ts);
  const std::size_t D = (std::size_t)_cfg.hidden;
  const std::size_t n4 = 4 * n;
  const std::size_t pv = 4 * (std::size_t)_cfg.in_channels;
  sc->n = (int)n;
  auto mk = [&](std::size_t bytes) {
    return _mc->make_shared_buffer(std::max<std::size_t>(bytes, 16));
  };
  sc->lqr = mk(n4 * (std::size_t)_cfg.lq_channels() * 2);
  sc->lq1 = mk(n4 * (std::size_t)_cfg.lq_hidden1 * 2);
  sc->lq2 = mk(n * (std::size_t)_cfg.lq_hidden2 * 2);
  sc->prow = mk(n * pv * 2);
  sc->x = mk(n * D * 2);
  sc->y = mk(n * D * 2);
  sc->q = mk(n * D * 2);
  sc->a = mk(n * D * 2);
  sc->o = mk(n * D * 2);
  sc->ff = mk(n * (std::size_t)_cfg.ffn * 2);
  sc->kr = mk((std::size_t)_cfg.context_frames * n * D * 2);
  sc->head = mk(n * pv * 2);
  sc->noise = mk(n4 * (std::size_t)_cfg.in_channels * 4);
  bool ok = true;
  for (const SharedBuffer* b : {&sc->lqr, &sc->lq1, &sc->lq2, &sc->prow,
                                &sc->x, &sc->y, &sc->q, &sc->a, &sc->o,
                                &sc->ff, &sc->kr, &sc->head, &sc->noise}) {
    ok = ok && !b->empty();
  }
  sc->ring_k.resize((std::size_t)_cfg.layers);
  sc->ring_v.resize((std::size_t)_cfg.layers);
  for (int i = 0; ok && i < _cfg.layers; ++i) {
    sc->ring_k[(std::size_t)i] =
        mk((std::size_t)_cfg.context_frames * n * D * 2);
    sc->ring_v[(std::size_t)i] =
        mk((std::size_t)_cfg.context_frames * n * D * 2);
    ok = !sc->ring_k[(std::size_t)i].empty() &&
         !sc->ring_v[(std::size_t)i].empty();
  }
  if (!ok) {
    if (err != nullptr) {
      *err = fmt("refvsr-wan: cannot allocate a {}x{} keyframe's ring and "
                 "working set ({} + {} MB)", pw, ph,
                 ring_bytes(_cfg, ph, pw) >> 20,
                 scratch_bytes(_cfg, ph, pw) >> 20)();
    }
    return false;
  }
  _sc = std::move(sc);
  _frame = 0;   // a new geometry is a new stream
  return true;
}

// ---- the ANE tier -----------------------------------------------------------

bool
MetalRelayVsrWan::ane_setup_(int rows)
{
  if (_ane_tried) { return _ane != nullptr; }
  _ane_tried = true;
  if (_mc->session() == nullptr || rows <= 0) { return false; }
  // Nothing to stage where an int8 tier holds every feed-forward: no
  // module is built, so none is held.
  bool any = false;
  for (const Block& b : _blocks) {
    any = any || (!b.f1.held() && !b.f2.held());
  }
  if (!any) {
    _mc->session()->info(
        fmt("relayvsr-keyframe: the int8 tier holds every block's "
            "feed-forward, which the ANE cannot stage; it stays on the "
            "GPU"));
    return false;
  }
  AneFeedForward::Options o;
  o.session = _mc->session();
  o.mc      = _mc;
  o.tag     = "relayvsr-keyframe";
  o.hidden  = _cfg.hidden;
  o.ffn     = _cfg.ffn;
  o.seq     = rows;
  o.rows    = _cfg.ane_rows;
  o.chunk   = relayvsr::ane_chunk(rows);
  o.gelu    = true;
  o.biases  = true;
  o.profile = std::getenv("VPIPE_RELAYVSR_ANE_PROFILE") != nullptr;
  _ane = AneFeedForward::create(o);
  return _ane != nullptr;
}

bool
MetalRelayVsrWan::ane_eligible_(int bi) const
{
  if (_ane == nullptr) { return false; }
  // A feed-forward held in the int8 layout is not a source the staging
  // reads: the GPU keeps it.
  const Block& b = _blocks[(std::size_t)bi];
  return !b.f1.held() && !b.f2.held();
}

void
MetalRelayVsrWan::ane_stage_(int bi)
{
  const Block& b = _blocks[(std::size_t)bi];
  auto src = [](const relayvsr::Proj& p) {
    AneFfnSource s;
    s.bias = &p.b;
    if (p.stored()) {
      // ComfyUI's int8 as stored: staged as the GPU's widening holds it.
      s.w      = &p.i8;
      s.scales = &p.i8s;
      s.bits   = AneFfnSource::kInt8;
      s.rot    = p.rot;
    } else {
      s.w = &p.w;
    }
    return s;
  };
  // A GELU module: {up, down}.
  _ane->stage(bi, std::vector<AneFfnSource>{src(b.f1), src(b.f2)}, 0);
}

bool
MetalRelayVsrWan::keyframe(const SharedBuffer& lr, int ph, int pw,
                           const std::vector<float>& noise,
                           SharedBuffer* latent, std::string* err,
                           const std::function<bool()>& stop)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  // h4 x w4 the LATENT grid (1/4 of the frame at 4x, 1/8 at 2x), h8 x w8
  // the token grid, half of it.
  const int ls = _cfg.latent_stride(), ts = _cfg.token_stride();
  if (ph <= 0 || pw <= 0 || ph % ts != 0 || pw % ts != 0) {
    return fail(fmt("refvsr-wan: the frame must be padded to a multiple of "
                    "{}", ts)());
  }
  const int C = _cfg.in_channels;
  const int h4 = ph / ls, w4 = pw / ls, h8 = h4 / 2, w8 = w4 / 2;
  if (noise.size() != (std::size_t)C * h4 * w4) {
    return fail(fmt("refvsr-wan: noise is not [48, ph/{}, pw/{}]", ls,
                    ls)());
  }
  if (!ensure_scratch_(ph, pw, err)) { return false; }
  Scratch& s = *_sc;
  // The accurate tier's split-K tuning of the shapes the last keyframe met,
  // with no encoder open.
  _ops->tune_pending();
  const relayvsr::Ops& op = *_ops;
  const long i8_0 = op.i8_gemms(), dense_0 = op.dense_gemms();
  const long cr_0 = op.convrot_gemms(), wid_0 = op.widened_gemms();
  const int n = s.n, n4 = h4 * w4, D = _cfg.hidden, NH = _cfg.heads;
  const int PV = 4 * C, W = _cfg.context_frames;
  const std::size_t lat_bytes = (std::size_t)C * n4 * 4;
  if (latent->empty() || latent->byte_size() < lat_bytes) {
    *latent = _mc->make_shared_buffer(lat_bytes);
    if (latent->empty()) { return fail("latent allocation failed"); }
  }
  std::memcpy(s.noise.contents(), noise.data(), lat_bytes);
  if (!_have_stream) {
    _stream = _mc->make_command_stream();
    _have_stream = true;
  }
  auto commit = [&](const char* what) {
    std::string ge;
    if (!_stream.commit().wait_ok(&ge)) {
      return fail(fmt("refvsr-wan: {} failed: {}", what, ge)());
    }
    return true;
  };
  relayvsr::Profile prof("keyframe");

  // ---- the frame's rows and the noise's patch rows, summed.
  {
    ComputeEncoder enc = _stream.begin_compute();
    const int lc = _cfg.lq_channels();
    enc.set_function(op.lq_rows);
    enc.set_buffer(0, lr); enc.set_buffer(1, s.lqr);
    enc.set_constant(2, ph); enc.set_constant(3, pw);
    enc.set_constant(4, _cfg.upscale);
    enc.dispatch({(unsigned)lc, (unsigned)n4, 1}, {64, 1, 1});
    op.gemm(enc, s.lqr, 0, _lq_c1w, &_lq_c1b, s.lq1, 0, n4,
            _cfg.lq_hidden1, lc);
    enc.set_function(op.rms_silu);
    enc.set_buffer(0, s.lq1); enc.set_buffer(1, _lq_n1);
    enc.set_buffer(2, s.lq1);
    enc.set_constant(3, _cfg.lq_hidden1); enc.set_constant(4, 1e-6f);
    enc.dispatch({256, (unsigned)n4, 1}, {256, 1, 1});
    op.conv3x3(enc, s.lq1, _lq_c2w, _lq_c2b, s.lq2, h4, w4, _cfg.lq_hidden1,
               _cfg.lq_hidden2, 2, false);
    enc.set_function(op.rms_silu);
    enc.set_buffer(0, s.lq2); enc.set_buffer(1, _lq_n2);
    enc.set_buffer(2, s.lq2);
    enc.set_constant(3, _cfg.lq_hidden2); enc.set_constant(4, 1e-6f);
    enc.dispatch({256, (unsigned)n, 1}, {256, 1, 1});
    op.gemm(enc, s.lq2, 0, _lq_ow, &_lq_ob, s.y, 0, n, D, _cfg.lq_hidden2);
    enc.set_function(op.patch_rows);
    enc.set_buffer(0, s.noise); enc.set_buffer(1, s.prow);
    enc.set_constant(2, C); enc.set_constant(3, h4); enc.set_constant(4, w4);
    enc.dispatch({(unsigned)PV, (unsigned)n, 1}, {64, 1, 1});
    op.gemm(enc, s.prow, 0, _patch_w, &_patch_b, s.x, 0, n, D, PV);
    enc.set_function(op.resadd);
    enc.set_buffer(0, s.x); enc.set_buffer(1, s.y); enc.set_buffer(2, s.x);
    enc.set_constant(3, n * D);
    enc.dispatch({(unsigned)(n * D), 1, 1}, {256, 1, 1});
    prof.mark(enc, "source");
    enc.end();
    if (!commit("the source rows")) { return false; }
  }
  if (_keep_debug) {
    _dbg_lq.resize((std::size_t)n * D);
    const auto* p = (const std::uint16_t*)s.y.contents();
    for (std::size_t i = 0; i < _dbg_lq.size(); ++i) {
      _dbg_lq[i] = from_bf16(p[i]);
    }
  }

  // ---- positions: the query at f - origin, slot s's key at its own
  // keyframe - origin (fixed_end), the slots in their physical order.
  const int f = _frame;
  const int valid = std::min(f + 1, W);
  const int origin = std::max(0, f - _cfg.rope_frames + 1);
  const int slot = f % W;
  std::vector<int> kpos((std::size_t)valid);
  for (int sl = 0; sl < valid; ++sl) {
    kpos[(std::size_t)sl] = (f - ((f - sl) % W)) - origin;
  }
  const std::size_t plane = (std::size_t)n * D * 2;   // bytes a slot
  auto rope = [&](ComputeEncoder& enc, const SharedBuffer& in,
                  std::size_t in_off, const SharedBuffer& out,
                  std::size_t out_off, int t) {
    enc.set_function(op.rope_wan);
    enc.set_buffer(0, in, in_off); enc.set_buffer(1, out, out_off);
    enc.set_buffer(2, _rope_cos); enc.set_buffer(3, _rope_sin);
    enc.set_constant(4, D); enc.set_constant(5, D);
    enc.set_constant(6, w8); enc.set_constant(7, t);
    enc.set_constant(8, NH);
    enc.dispatch({64, (unsigned)n, (unsigned)NH}, {64, 1, 1});
  };
  auto rms = [&](ComputeEncoder& enc, const SharedBuffer& x,
                 std::size_t off, const SharedBuffer& g, int rows) {
    enc.set_function(op.rms);
    enc.set_buffer(0, x, off); enc.set_buffer(1, g);
    enc.set_buffer(2, x, off);
    enc.set_constant(3, D); enc.set_constant(4, _cfg.eps);
    enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
  };
  auto ln_mod = [&](ComputeEncoder& enc, const SharedBuffer& mod, int which) {
    // which 0: shift 0 / scale 1; which 1: shift 3 / scale 4.
    enc.set_function(op.ln_mod);
    enc.set_buffer(0, s.x);
    enc.set_buffer(1, mod, (std::size_t)(which ? 4 : 1) * D * 4);
    enc.set_buffer(2, mod, (std::size_t)(which ? 3 : 0) * D * 4);
    enc.set_buffer(3, s.y);
    enc.set_constant(4, D); enc.set_constant(5, _cfg.eps);
    enc.dispatch({256, (unsigned)n, 1}, {256, 1, 1});
  };
  auto gated = [&](ComputeEncoder& enc, const SharedBuffer& mod, int which,
                   const SharedBuffer& sub) {
    enc.set_function(op.gated_res);
    enc.set_buffer(0, s.x);
    enc.set_buffer(1, mod, (std::size_t)(which ? 5 : 2) * D * 4);
    enc.set_buffer(2, sub);
    enc.set_constant(3, D);
    enc.set_constant(4, (unsigned)((std::size_t)n * D));
    enc.dispatch({(unsigned)((std::size_t)n * D), 1, 1}, {256, 1, 1});
  };

  using Clk = std::chrono::steady_clock;
  auto ms_since = [](Clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clk::now() - t0).count();
  };
  const int F = _cfg.ffn;
  const bool ane_on = _cfg.ane_ffn && ane_setup_(n);
  _ane_rows = 0;
  if (_ane) { _ane->forget_staged(); }
  for (int bi = 0; bi < _cfg.layers; ++bi) {
    if (stop && stop()) {
      if (_ane) { _ane->join(); }
      return false;
    }
    const Block& b = _blocks[(std::size_t)bi];
    const SharedBuffer& rk = s.ring_k[(std::size_t)bi];
    const SharedBuffer& rv = s.ring_v[(std::size_t)bi];
    // Whether this block's feed-forward splits with the Neural Engine: asked
    // at its start, so its weights are staged under its attention.
    const AneFeedForward::Plan plan =
        ane_on && ane_eligible_(bi) ? _ane->plan_block()
                                    : AneFeedForward::Plan::kGpu;
    const bool ane_block = plan == AneFeedForward::Plan::kSplit;
    const bool ane_probe = plan == AneFeedForward::Plan::kProbe;
    if (ane_block) { ane_stage_(bi); }
    ComputeEncoder enc = _stream.begin_compute();
    // Self-attention: this keyframe's K and V go straight into its slot.
    ln_mod(enc, b.mod, 0);
    op.proj_shared(enc, s.y, 0, n,
                   {{&b.q, &s.q, 0},
                    {&b.k, &rk, (std::size_t)slot * plane},
                    {&b.v, &rv, (std::size_t)slot * plane}});
    rms(enc, s.q, 0, b.nq, n);
    rms(enc, rk, (std::size_t)slot * plane, b.nk, n);
    prof.mark(enc, "qkv");
    rope(enc, s.q, 0, s.q, 0, f - origin);
    std::vector<relayvsr::WindowPlane> planes((std::size_t)valid);
    for (int sl = 0; sl < valid; ++sl) {
      rope(enc, rk, (std::size_t)sl * plane, s.kr, (std::size_t)sl * plane,
           kpos[(std::size_t)sl]);
      relayvsr::WindowPlane& pl = planes[(std::size_t)sl];
      pl.k = &s.kr; pl.k_off = (std::size_t)sl * plane; pl.k_tok = D;
      pl.v = &rv; pl.v_off = (std::size_t)sl * plane; pl.v_tok = D;
      pl.win_h = _cfg.window_h; pl.win_w = _cfg.window_w;
      pl.shifted = true;
    }
    prof.mark(enc, "rope");
    op.window_attention(enc, s.q, 0, D, planes, s.a, 0, D, h8, w8, NH, NH,
                        _cfg.head_dim, _cfg.sage_mode());
    prof.mark(enc, "self-attn");
    op.proj(enc, s.a, 0, b.o, s.o, 0, n);
    gated(enc, b.mod, 0, s.o);
    prof.mark(enc, "o-proj");
    // Cross-attention onto the folded context.
    enc.set_function(op.ln_affine);
    enc.set_buffer(0, s.x); enc.set_buffer(1, b.n2w);
    enc.set_buffer(2, b.n2b); enc.set_buffer(3, s.y);
    enc.set_constant(4, D); enc.set_constant(5, _cfg.eps);
    enc.dispatch({256, (unsigned)n, 1}, {256, 1, 1});
    op.proj(enc, s.y, 0, b.xq, s.q, 0, n);
    rms(enc, s.q, 0, b.xnq, n);
    prof.mark(enc, "cross-q");
    op.cross_attention(enc, s.q, n, b.ck, b.cv, _cfg.text_tokens, s.a, NH,
                       _cfg.sage_mode());
    prof.mark(enc, "cross-attn");
    op.proj(enc, s.a, 0, b.xo, s.o, 0, n);
    enc.set_function(op.resadd);
    enc.set_buffer(0, s.x); enc.set_buffer(1, s.o); enc.set_buffer(2, s.x);
    enc.set_constant(3, n * D);
    enc.dispatch({(unsigned)(n * D), 1, 1}, {256, 1, 1});
    prof.mark(enc, "cross-o");
    // The feed-forward. Split with the Neural Engine, the ANE takes the
    // LAST rows of y into the same rows of o (its biases staged with it):
    // the attention is committed and drained first -- the ANE reads y on
    // the host -- and the GPU's rows encoded only once it has started.
    ln_mod(enc, b.mod, 1);
    int a_rows = 0;
    double drain_ms = 0.0;
    if (ane_block || ane_probe) {
      const auto t0 = Clk::now();
      enc.end();
      metal_compute::CommandStream::Fence fence = _stream.commit();
      const bool staged = ane_block && _ane->join_stage(bi);
      std::string ge;
      if (!fence.wait_ok(&ge)) {
        if (_ane) { _ane->join(); }
        return fail("refvsr-wan: a block failed: " + ge);
      }
      drain_ms = ms_since(t0);
      if (staged) {
        a_rows = _ane->begin(s.y, s.o, n);
      } else if (ane_block && _mc->session() != nullptr) {
        _mc->session()->warn(fmt("relayvsr-keyframe: staging block {}'s "
                                 "feed-forward for the ANE failed; it keeps "
                                 "the GPU", bi));
      }
      enc = _stream.begin_compute();
      prof.host("ff-drain");
    }
    // The GPU's rows [r0, r0 + m).
    auto gpu_ff = [&](int r0, int m, bool marks) {
      if (m <= 0) { return; }
      op.proj(enc, s.y, (std::size_t)r0 * D * 2, b.f1, s.ff,
              (std::size_t)r0 * F * 2, m);
      enc.set_function(op.gelu);
      enc.set_buffer(0, s.ff, (std::size_t)r0 * F * 2);
      enc.set_buffer(1, s.ff, (std::size_t)r0 * F * 2);
      enc.set_constant(2, m * F);
      enc.dispatch({(unsigned)(m * F), 1, 1}, {256, 1, 1});
      if (marks) { prof.mark(enc, "ff-up"); }
      op.proj(enc, s.ff, (std::size_t)r0 * F * 2, b.f2, s.o,
              (std::size_t)r0 * D * 2, m);
    };
    const auto tg0 = Clk::now();
    gpu_ff(0, n - a_rows, a_rows == 0 && !ane_probe);
    if (a_rows > 0 || ane_probe) {
      std::string ge;
      if (!metal_compute::CommandStream::sync_and_reopen(enc, &ge)) {
        if (_ane) { _ane->join(); }
        return fail("refvsr-wan: a block failed: " + ge);
      }
      const double gpu_ms = ms_since(tg0);
      if (a_rows > 0) {
        const bool ok = _ane->finish(bi, n, gpu_ms, drain_ms);
        _ane_rows += a_rows;
        // Rows the ANE did not write (an fp16 overflow it could not
        // rescale) are the GPU's again.
        const int lost = ok ? -1 : _ane->lost_row0();
        if (lost >= 0 && lost < n) { gpu_ff(lost, n - lost, false); }
      } else {
        _ane->note_probe(drain_ms, gpu_ms);
      }
      prof.host(a_rows > 0 ? "ff-split" : "ff-probe");
    }
    gated(enc, b.mod, 1, s.o);
    prof.mark(enc, a_rows > 0 || ane_probe ? "ff-tail" : "ff-down");
    enc.end();
    if (!commit("a block")) { return false; }
    if (_keep_debug && bi == 0) {
      _dbg_blk0.resize((std::size_t)n * D);
      const auto* p = (const std::uint16_t*)s.x.contents();
      for (std::size_t i = 0; i < _dbg_blk0.size(); ++i) {
        _dbg_blk0[i] = from_bf16(p[i]);
      }
    }
  }

  // ---- the head and the one step.
  {
    ComputeEncoder enc = _stream.begin_compute();
    enc.set_function(op.ln_mod);
    enc.set_buffer(0, s.x); enc.set_buffer(1, _head_scale);
    enc.set_buffer(2, _head_shift); enc.set_buffer(3, s.y);
    enc.set_constant(4, D); enc.set_constant(5, _cfg.eps);
    enc.dispatch({256, (unsigned)n, 1}, {256, 1, 1});
    op.gemm(enc, s.y, 0, _head_w, &_head_b, s.head, 0, n, PV, D);
    enc.set_function(op.x0);
    enc.set_buffer(0, s.head); enc.set_buffer(1, s.noise);
    enc.set_buffer(2, *latent);
    enc.set_constant(3, C); enc.set_constant(4, h4); enc.set_constant(5, w4);
    enc.dispatch({(unsigned)w4, (unsigned)h4, (unsigned)C}, {32, 8, 1});
    prof.mark(enc, "head");
    enc.end();
    if (!commit("the head")) { return false; }
  }
  _i8_gemms = op.i8_gemms() - i8_0;
  _dense_gemms = op.dense_gemms() - dense_0;
  _cr_gemms = op.convrot_gemms() - cr_0;
  _widened = op.widened_gemms() - wid_0;
  prof.report(_mc, fmt("{} rows, {} int8 / {} dense GEMMs ({} ConvRot as "
                       "stored, {} widened), {} ANE rows", n, _i8_gemms,
                       _dense_gemms, _cr_gemms, _widened, _ane_rows)());
  ++_frame;
  return true;
}

}  // namespace genai
}  // namespace vpipe
