#include "generative-models/quantize/model-quantizer.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/ui-delegate-intf.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/quantize/affine-quantizer.h"
#include "generative-models/quantize/safetensors-writer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vpipe::genai {

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

std::vector<std::string>
ModelQuantizer::default_quant_linears()
{
  return {"q_proj", "k_proj", "v_proj", "o_proj",
          "gate_proj", "up_proj", "down_proj",
          // Qwen3.5 gated-DeltaNet linears (hybrid layers).
          "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj"};
}

namespace {

// Leaf segment just before a trailing ".weight" (e.g.
// "...self_attn.q_proj.weight" -> "q_proj"). Empty if not a ".weight".
std::string
weight_leaf_(const std::string& name)
{
  static const std::string kSuf = ".weight";
  if (name.size() <= kSuf.size() ||
      name.compare(name.size() - kSuf.size(), kSuf.size(), kSuf) != 0) {
    return {};
  }
  const std::string pfx = name.substr(0, name.size() - kSuf.size());
  const auto dot = pfx.rfind('.');
  return dot == std::string::npos ? pfx : pfx.substr(dot + 1);
}

bool
read_file_(const std::string& path, std::string* out)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { return false; }
  std::ostringstream ss;
  ss << in.rdbuf();
  *out = ss.str();
  return true;
}

bool
write_file_(const std::string& path, const std::string& data)
{
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) { return false; }
  out.write(data.data(), (std::streamsize)data.size());
  return out.good();
}

inline float bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f; std::memcpy(&f, &u, 4); return f;
}
inline std::uint16_t f32_to_bf16_(float f)
{
  std::uint32_t x; std::memcpy(&x, &f, 4);
  const std::uint32_t r = x + 0x7fffu + ((x >> 16) & 1u);
  return (std::uint16_t)(r >> 16);
}

// AWQ per-input-channel scale search. Over an alpha grid, builds the protect
// scale s_c = act_c^alpha (normalized to unit mean, clamped) and measures the
// activation-weighted reconstruction error of group-affine-quantizing the
// matrices that share this input (W -> W*diag(s) for inv=false [qkv/gate-up,
// folded into the upstream norm], or W -> W*diag(1/s) for inv=true [down,
// folded into up rows]); picks the alpha minimizing total error. Host f32,
// row-subsampled (stride) for speed. Fills best_s [K].
void
awq_search(const float* act, int K,
           const std::vector<std::pair<const float*, int>>& mats, int bits,
           int group, bool inv, int row_stride, std::vector<float>& best_s)
{
  static const float grid[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
  const int qmax = (1 << bits) - 1;
  std::vector<float> s((std::size_t)K);
  best_s.assign((std::size_t)K, 1.0f);
  double best = 1e300;
  for (float alpha : grid) {
    double m = 0.0;
    for (int c = 0; c < K; ++c) {
      s[(std::size_t)c] = std::pow(std::max(act[c], 1e-8f), alpha);
      m += s[(std::size_t)c];
    }
    m = (m > 0.0) ? m / K : 1.0;
    for (int c = 0; c < K; ++c) {
      float v = (float)(s[(std::size_t)c] / m);
      s[(std::size_t)c] = std::min(std::max(v, 0.1f), 10.0f);
    }
    double err = 0.0;
    for (const auto& mw : mats) {
      const float* W = mw.first;
      const int N = mw.second;
      for (int n = 0; n < N; n += row_stride) {
        const float* r = W + (std::size_t)n * K;
        for (int g0 = 0; g0 < K; g0 += group) {
          float mn = 1e30f, mx = -1e30f;
          for (int c = g0; c < g0 + group; ++c) {
            const float wp = r[c] * (inv ? 1.0f / s[(std::size_t)c]
                                         : s[(std::size_t)c]);
            mn = std::min(mn, wp); mx = std::max(mx, wp);
          }
          const float sc = (mx - mn) / (float)qmax;
          const float isc = sc > 0.0f ? 1.0f / sc : 0.0f;
          for (int c = g0; c < g0 + group; ++c) {
            const float sscl = s[(std::size_t)c];
            const float wp = r[c] * (inv ? 1.0f / sscl : sscl);
            int q = (int)std::lround((wp - mn) * isc);
            q = std::min(std::max(q, 0), qmax);
            const float dq = (float)q * sc + mn;
            const float eff = dq * (inv ? sscl : 1.0f / sscl);
            const float e = (r[c] - eff) * act[c];
            err += (double)e * e;
          }
        }
      }
    }
    if (err < best) { best = err; best_s = s; }
  }
}

// AWQ per-group weight clip (paired with awq_search's scale). For each group,
// pick the clip ratio in `grid` that minimizes the activation-weighted
// group-affine reconstruction error, then shrink the group's values toward the
// group midpoint by that ratio IN PLACE (v -> mid + (v-mid)*c, which saturates
// outliers) -- exactly the kernel's clip form, so the clamped weights quantize
// correctly at clip=1. act = per-input-channel magnitude [K]. Every group is
// searched + applied (no subsample); independent per group.
void
awq_clip_search(float* W, int N, int K, int bits, int group, const float* act)
{
  static const float grid[] = {1.0f, 0.9f, 0.8f, 0.7f, 0.6f};
  const int qmax = (1 << bits) - 1;
  for (int n = 0; n < N; ++n) {
    float* r = W + (std::size_t)n * K;
    for (int g0 = 0; g0 < K; g0 += group) {
      float mn = 1e30f, mx = -1e30f;
      for (int c = g0; c < g0 + group; ++c) {
        mn = std::min(mn, r[c]); mx = std::max(mx, r[c]);
      }
      const float mid = 0.5f * (mn + mx);
      double best = 1e300; float best_c = 1.0f;
      for (float cl : grid) {
        const float lo = mid + (mn - mid) * cl;
        const float hi = mid + (mx - mid) * cl;
        const float sc = (hi - lo) / (float)qmax;
        const float isc = sc > 0.0f ? 1.0f / sc : 0.0f;
        double err = 0.0;
        for (int c = g0; c < g0 + group; ++c) {
          const float wc = std::min(std::max(r[c], lo), hi);
          int q = (int)std::lround((wc - lo) * isc);
          q = std::min(std::max(q, 0), qmax);
          const float dq = (float)q * sc + lo;
          const float e = (r[c] - dq) * act[c];
          err += (double)e * e;
        }
        if (err < best) { best = err; best_c = cl; }
      }
      if (best_c < 1.0f) {
        const float lo = mid + (mn - mid) * best_c;
        const float hi = mid + (mx - mid) * best_c;
        for (int c = g0; c < g0 + group; ++c) {
          r[c] = std::min(std::max(r[c], lo), hi);
        }
      }
    }
  }
}

// Relative group-affine reconstruction error of W[N,K] at `bits`/`group`:
// sqrt(sum_sq(w - dequant(quant(w))) / sum_sq(w)) over rows subsampled by
// `row_stride`. Plain asymmetric min/max (no AWQ scale) -- a sensitivity
// proxy for the mixed-precision bit assignment.
float
recon_err_(const float* W, int N, int K, int bits, int group, int row_stride)
{
  const int qmax = (1 << bits) - 1;
  double se = 0.0, sw = 0.0;
  for (int n = 0; n < N; n += row_stride) {
    const float* r = W + (std::size_t)n * K;
    for (int g0 = 0; g0 < K; g0 += group) {
      float mn = 1e30f, mx = -1e30f;
      for (int c = g0; c < g0 + group; ++c) {
        mn = std::min(mn, r[c]); mx = std::max(mx, r[c]);
      }
      const float sc = (mx - mn) / (float)qmax;
      const float isc = sc > 0.0f ? 1.0f / sc : 0.0f;
      for (int c = g0; c < g0 + group; ++c) {
        int q = (int)std::lround((r[c] - mn) * isc);
        q = std::min(std::max(q, 0), qmax);
        const float dq = (float)q * sc + mn;
        const float e = r[c] - dq;
        se += (double)e * e; sw += (double)r[c] * r[c];
      }
    }
  }
  return sw > 0.0 ? (float)std::sqrt(se / sw) : 0.0f;
}

// Load a [n_layers, channels] f32 calibration array.
std::vector<float>
load_calib_(const std::string& dir, const std::string& tap, int nL, int ch)
{
  std::ifstream in((std::filesystem::path(dir) / ("calib_" + tap + ".f32"))
                       .string(), std::ios::binary);
  std::vector<float> v;
  if (!in) { return v; }
  v.resize((std::size_t)nL * ch);
  in.read(reinterpret_cast<char*>(v.data()),
          (std::streamsize)v.size() * 4);
  if (!in) { v.clear(); }
  return v;
}

// Throttled in-place progress bar -- redraws on a carriage-return only
// when the integer percentage changes (the frame is space-padded so a
// shorter redraw fully overwrites a longer prior one).
void quant_progress_(vpipe::UiTextStream* bar, const char* tag, int done,
                     int total, int& last_pct)
{
  if (bar == nullptr || total <= 0) { return; }
  int pct = static_cast<int>(static_cast<long>(done) * 100 / total);
  if (pct < 0) { pct = 0; } else if (pct > 100) { pct = 100; }
  if (pct == last_pct) { return; }
  last_pct = pct;
  constexpr int W = 24;
  const int fill = pct * W / 100;
  std::string b(static_cast<std::size_t>(fill), '#');
  b += std::string(static_cast<std::size_t>(W - fill), '-');
  std::string line = fmt("\r[{}] {}% {} ({}/{})", b, pct, tag, done,
                         total)();
  while (line.size() < 64) { line += ' '; }   // wipe stale tail
  bar->write(line);
}

}  // namespace

bool
ModelQuantizer::run(const std::string& in_dir, const std::string& out_dir,
                    const QuantizeOptions& opt, std::string* err,
                    const std::function<bool()>& stop) const
{
  namespace fs = std::filesystem;
  auto fail = [&](const std::string& m) { if (err) { *err = m; }
                                          return false; };

  if (_mc == nullptr) { return fail("model-quantize: null MetalCompute"); }
  if ((opt.bits != 4 && opt.bits != 8) ||
      (opt.group != 32 && opt.group != 64)) {
    return fail("model-quantize: bits must be 4|8, group 32|64");
  }
  if (opt.mixed && (opt.bits != 4 || opt.high_bits != 8 || opt.group != 64)) {
    // The mixed-affine decode kernels are w4g64 + w8g64 only.
    return fail("model-quantize: mixed requires bits=4 high_bits=8 group=64");
  }
  const SessionContextIntf* S = _mc->session();
  std::unique_ptr<vpipe::UiTextStream> bar;
  if (S) { bar = S->open_text_stream(); }
  AffineQuantizer q(_mc);
  if (!q.valid()) {
    return fail("model-quantize: affine quant kernels unavailable");
  }
  auto src = MetalLlamaWeights::open_model(in_dir);
  if (!src.has_value()) {
    return fail("model-quantize: cannot open source model: " + in_dir);
  }

  std::error_code ec;
  fs::create_directories(out_dir, ec);

  std::unordered_set<std::string> quant_set;
  for (const auto& s :
       (opt.quant_linears.empty() ? default_quant_linears()
                                  : opt.quant_linears)) {
    quant_set.insert(s);
  }
  // MLX-convention: quantize the embedding table + (untied) lm_head so a
  // standard LM reloads through the affine path. The plain pass handles them
  // (not per-layer, so the SmoothQuant/mixed passes skip them).
  if (opt.quant_embeddings) {
    quant_set.insert("embed_tokens");
    quant_set.insert("lm_head");
  }

  // Deterministic tensor order.
  std::vector<std::string> names = src->tensor_names();
  std::sort(names.begin(), names.end());
  if (S) {
    S->log_normal(fmt(
        "model-quantize: {} tensors -> {}-bit g{} ({} -> {})",
        names.size(), opt.bits, opt.group, in_dir, out_dir));
  }

  SafetensorsWriter wr(out_dir, opt.shard_max_bytes);
  int n_quant = 0, n_pass = 0;
  std::unordered_set<std::string> handled;   // names done by the SQ pass

  // ---- helpers (host f32) shared by the SmoothQuant pass ------------------
  auto load_f32 = [&](const std::string& nm, int& N, int& K)
      -> std::vector<float> {
    std::vector<float> out;
    const auto* ti = src->info(nm);
    SharedBuffer raw = src->load(nm, _mc);
    if (ti == nullptr || raw.empty()) { return out; }
    N = (int)ti->shape[0];
    K = ti->shape.size() > 1 ? (int)ti->shape[1] : 1;
    const std::size_t n = (std::size_t)N * K;
    out.resize(n);
    if (ti->dtype == "BF16") {
      const auto* s = static_cast<const std::uint16_t*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { out[i] = bf16_to_f32_(s[i]); }
    } else if (ti->dtype == "F16") {
      const auto* s = static_cast<const _Float16*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { out[i] = (float)s[i]; }
    } else {
      out.clear();
    }
    return out;
  };
  auto scale_cols = [](std::vector<float>& W, int N, int K,
                       const std::vector<float>& s, bool inv) {
    for (int n = 0; n < N; ++n) {
      float* r = W.data() + (std::size_t)n * K;
      for (int c = 0; c < K; ++c) {
        r[c] *= inv ? (1.0f / s[(std::size_t)c]) : s[(std::size_t)c];
      }
    }
  };
  auto scale_rows = [](std::vector<float>& W, int N, int K,
                       const std::vector<float>& s) {
    for (int n = 0; n < N; ++n) {
      float* r = W.data() + (std::size_t)n * K;
      const float sn = s[(std::size_t)n];
      for (int c = 0; c < K; ++c) { r[c] *= sn; }
    }
  };
  auto quant_write = [&](const std::string& pfx, const std::vector<float>& W,
                         int N, int K, int bits) -> bool {
    SharedBuffer in = _mc->make_shared_buffer((std::size_t)N * K * 2);
    if (in.empty()) { return false; }
    auto* d = static_cast<_Float16*>(in.contents());
    for (std::size_t i = 0; i < W.size(); ++i) { d[i] = (_Float16)W[i]; }
    SharedBuffer w, s, b;
    if (!q.quantize_linear(in, false, N, K, bits, opt.group, opt.clip,
                           w, s, b)) {
      return false;
    }
    const std::int64_t wcols = (std::int64_t)K * bits / 32;
    const std::int64_t gcols = (std::int64_t)K / opt.group;
    return wr.add(pfx + ".weight", "U32", {N, wcols}, w.contents(),
                  w.byte_size()) &&
           wr.add(pfx + ".scales", "F16", {N, gcols}, s.contents(),
                  s.byte_size()) &&
           wr.add(pfx + ".biases", "F16", {N, gcols}, b.contents(),
                  b.byte_size());
  };
  auto norm_write = [&](const std::string& nm, const std::vector<float>& v)
      -> bool {
    SharedBuffer buf = _mc->make_shared_buffer(v.size() * 2);
    if (buf.empty()) { return false; }
    auto* d = static_cast<std::uint16_t*>(buf.contents());
    for (std::size_t i = 0; i < v.size(); ++i) { d[i] = f32_to_bf16_(v[i]); }
    return wr.add(nm, "BF16", {(std::int64_t)v.size()}, buf.contents(),
                  buf.byte_size());
  };
  // Zero-centered RMSNorm (raw-HF Qwen3.5/3.6): which norm weights need the
  // +1 fold so vpipe's plain-`weight` kernel reproduces the model's
  // (1 + weight). Excludes the gated GDN norm (linear_attn.norm, ones-init).
  auto norm_plus_one = [&](const std::string& name) -> bool {
    if (!opt.norm_offset) { return false; }
    auto ends = [&](const std::string& suf) {
      return name.size() >= suf.size() &&
             name.compare(name.size() - suf.size(), suf.size(), suf) == 0;
    };
    if (ends("linear_attn.norm.weight")) { return false; }   // gated, as-is
    // MTP-head norms are stored RAW (zero-centered) and get the +1 at load
    // (the MTP loader mirrors the OptiQ sibling convention), so don't fold
    // them here -- folding would double the +1.
    if (name.rfind("mtp.", 0) == 0 ||
        name.find(".mtp.") != std::string::npos) { return false; }
    return ends("input_layernorm.weight") ||
           ends("post_attention_layernorm.weight") ||
           ends("q_norm.weight") || ends("k_norm.weight") ||
           ends(".norm.weight");                              // final model norm
  };

  // ---- mixed-precision bit map (unsloth-style sensitivity quant) ----------
  // Default: every quantized tensor takes opt.bits. When opt.mixed, a pre-pass
  // ranks quantizable linears by how much promotion to high_bits HELPS (the
  // base->high reconstruction-error drop) and promotes the top mixed_frac.
  std::unordered_map<std::string, int> bit_map;
  auto tbits = [&](const std::string& name) -> int {
    auto it = bit_map.find(name);
    return it == bit_map.end() ? opt.bits : it->second;
  };
  if (opt.mixed) {
    // Bit width is assigned per-LAYER (every quantizable linear in a layer
    // shares one width), NOT per-tensor: the runtime's mixed decode/prefill
    // path corrupts a layer whose projections carry DIFFERENT widths (the
    // fused q|k|v and gate|up groups assume a uniform width), and per-layer
    // assignment is also what unsloth / llama.cpp dynamic quant do. We rank
    // layers by aggregate sensitivity = sum over the layer's linears of how
    // much promotion base->high helps (the recon-error drop), then promote
    // the most-sensitive mixed_frac fraction of layers to high_bits.
    if (opt.n_layers <= 0) { return fail("mixed: n_layers required"); }
    std::vector<double> layer_gain((std::size_t)opt.n_layers, 0.0);
    std::vector<std::vector<std::string>> layer_names((std::size_t)opt.n_layers);
    for (const auto& name : names) {
      const auto* ti = src->info(name);
      if (ti == nullptr) { continue; }
      const std::string leaf = weight_leaf_(name);
      const bool fp = ti->dtype == "BF16" || ti->dtype == "F16";
      if (leaf.empty() || quant_set.count(leaf) == 0 ||
          ti->shape.size() != 2 || !fp ||
          ti->shape[1] % opt.group != 0 ||
          name.rfind(opt.layer_prefix, 0) != 0) {
        continue;
      }
      const int L = std::atoi(name.c_str() + opt.layer_prefix.size());
      if (L < 0 || L >= opt.n_layers) { continue; }
      int N, K;
      auto W = load_f32(name, N, K);
      if (W.empty()) { return fail("mixed: load failed: " + name); }
      const int RS = 8;   // row-subsample stride for the sensitivity estimate
      const float e_lo = recon_err_(W.data(), N, K, opt.bits, opt.group, RS);
      const float e_hi = recon_err_(W.data(), N, K, opt.high_bits, opt.group,
                                    RS);
      layer_gain[(std::size_t)L] += std::max(e_lo - e_hi, 0.0f);
      layer_names[(std::size_t)L].push_back(name);
    }
    std::vector<int> order((std::size_t)opt.n_layers);
    for (int L = 0; L < opt.n_layers; ++L) { order[(std::size_t)L] = L; }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return layer_gain[(std::size_t)a] > layer_gain[(std::size_t)b];
    });
    const std::size_t n_hi_l =
        (std::size_t)std::llround(opt.mixed_frac * (double)opt.n_layers);
    int n_hi_t = 0;
    for (std::size_t i = 0; i < order.size(); ++i) {
      const int b = i < n_hi_l ? opt.high_bits : opt.bits;
      for (const auto& nm : layer_names[(std::size_t)order[i]]) {
        bit_map[nm] = b;
        if (b == opt.high_bits) { ++n_hi_t; }
      }
    }
    if (_mc->session() != nullptr) {
      _mc->session()->info(fmt(
          "model-quantize: mixed bit map: {}/{} layers ({} linears) @ {}-bit, "
          "rest @ {}-bit", n_hi_l, opt.n_layers, n_hi_t, opt.high_bits,
          opt.bits));
    }
  }

  // True iff `name` ends with `suf`.
  auto ends_with = [](const std::string& name, const std::string& suf) {
    return name.size() >= suf.size() &&
           name.compare(name.size() - suf.size(), suf.size(), suf) == 0;
  };
  // Layer index parsed from a `<layer_prefix>N....` name (-1 if not a layer).
  auto layer_of = [&](const std::string& name) -> int {
    if (name.rfind(opt.layer_prefix, 0) != 0) { return -1; }
    const char* p = name.c_str() + opt.layer_prefix.size();
    if (*p < '0' || *p > '9') { return -1; }
    return std::atoi(p);
  };

  // ---- MoE per-expert AWQ fold state (raw-HF Qwen3.5/3.6 MoE) -------------
  // The experts, router (mlp.gate), shared expert, and shared-expert gate ALL
  // read one shared input -- the post_attention_layernorm output -- so a
  // single per-input-channel scale s[H] folds fp-equivalently (1/s into the
  // norm, s into every consumer's input columns). The per-expert down input
  // (silu(gate_e)*up_e) takes its own scale s_e[I] (fold into up_e rows +
  // 1/s_e into down_e cols). s[H] is computed in the SmoothQuant pass (where
  // post_attention_layernorm is folded + written); s_e[I] is computed in the
  // main loop's down_proj branch (where the down weights are loaded). Both
  // are stashed here and reused so Y stays fp-exact across the split.
  std::unordered_map<int, std::vector<float>> moe_s_in;     // L -> s[H]
  std::unordered_map<int, std::vector<float>> moe_s_down;   // L -> [E*I]
  // Per-expert calib (small: gateup [nL,E*H] ~2 MB/layer, down [nL,E*I]
  // ~0.5 MB/layer at E=256,H=2048,I=512). Loaded once, indexed per layer in
  // both passes; the shared input magnitude reuses the dense calib_gateup.
  std::vector<float> moe_cgu, moe_ceg, moe_ced;
  int moe_E = 0, moe_I = 0, moe_H = 0;
  if (opt.smoothquant && opt.n_layers > 0) {
    for (const auto& name : names) {
      if (!ends_with(name, ".mlp.experts.gate_up_proj")) { continue; }
      const auto* ti = src->info(name);
      if (ti != nullptr && ti->shape.size() == 3) {
        moe_E = (int)ti->shape[0];
        moe_I = (int)ti->shape[1] / 2;
        moe_H = (int)ti->shape[2];
      }
      break;
    }
    if (moe_E > 0 && !opt.calib_dir.empty()) {
      moe_cgu = load_calib_(opt.calib_dir, "gateup", opt.n_layers, moe_H);
      moe_ceg = load_calib_(opt.calib_dir, "expert_gateup", opt.n_layers,
                            moe_E * moe_H);
      moe_ced = load_calib_(opt.calib_dir, "expert_down", opt.n_layers,
                            moe_E * moe_I);
      if (moe_cgu.empty() || moe_ceg.empty() || moe_ced.empty()) {
        return fail("smoothquant: MoE per-expert calib (gateup / "
                    "expert_gateup / expert_down) missing in " +
                    opt.calib_dir);
      }
    }
  }

  // ---- SmoothQuant layer-aware fold pass ----------------------------------
  if (opt.smoothquant) {
    if (opt.n_layers <= 0) { return fail("smoothquant: n_layers required"); }
    const int nL = opt.n_layers;
    // H from layer-0's input_layernorm (present on every layer, full-attn /
    // GDN / MoE). FFN from the first DENSE-MLP layer's down_proj [H,FFN]; it
    // stays 0 when the whole stack is MoE (no mlp.gate_proj) -- the standard-
    // MLP fold + its gateup/down calibration are then skipped per layer.
    const std::string p0 = opt.layer_prefix + "0.";
    const auto* iln0 = src->info(p0 + "input_layernorm.weight");
    if (iln0 == nullptr || iln0->shape.empty()) {
      return fail("smoothquant: layer-0 input_layernorm missing");
    }
    const int H = (int)iln0->shape[0];
    int FFN = 0;
    for (int L = 0; L < nL && FFN == 0; ++L) {
      const auto* idn = src->info(opt.layer_prefix + std::to_string(L) +
                                  ".mlp.down_proj.weight");
      if (idn != nullptr && idn->shape.size() == 2) {
        FFN = (int)idn->shape[1];
      }
    }
    // o_proj input dim (attn-output width) for the o_proj clip, from the first
    // full-attention layer; GDN layers have no o_proj.
    int QD = 0;
    for (int L = 0; L < nL && QD == 0; ++L) {
      const auto* io = src->info(opt.layer_prefix + std::to_string(L) +
                                 ".self_attn.o_proj.weight");
      if (io != nullptr && io->shape.size() == 2) { QD = (int)io->shape[1]; }
    }
    const auto cq = load_calib_(opt.calib_dir, "qkv", nL, H);
    // gateup/down calib only feed the dense-MLP fold; skipped for a pure-MoE
    // stack (FFN==0), whose experts are plain-quantized in the main pass.
    const auto cg = FFN > 0 ? load_calib_(opt.calib_dir, "gateup", nL, H)
                            : std::vector<float>{};
    const auto cd = FFN > 0 ? load_calib_(opt.calib_dir, "down", nL, FFN)
                            : std::vector<float>{};
    if (cq.empty() || (FFN > 0 && (cg.empty() || cd.empty()))) {
      return fail("smoothquant: calib stats missing in " + opt.calib_dir);
    }
    // o_proj clip needs its own input (attn-output) calib; only required when
    // AWQ clipping is on AND the model has full-attention layers.
    const auto co = (opt.awq_clip && QD > 0)
                        ? load_calib_(opt.calib_dir, "o", nL, QD)
                        : std::vector<float>{};
    if (opt.awq_clip && QD > 0 && co.empty()) {
      return fail("awq_clip: calib_o.f32 missing in " + opt.calib_dir);
    }
    const int RS = 8;   // row-subsample stride for the AWQ error estimate

    // One quantized linear awaiting write: name + (folded) f32 weights + dims.
    struct Mat { std::string name; std::vector<float> W; int N = 0, K = 0; };

    if (S) {
      S->log_normal(fmt(
          "model-quantize: AWQ/SmoothQuant fold over {} layers", nL));
    }
    int fold_pct = -1;
    for (int L = 0; L < nL; ++L) {
      if (stop()) { return fail("model-quantize: stopped by request"); }
      if (S) { S->log_verbose(fmt("  fold layer {}/{}", L + 1, nL)); }
      quant_progress_(bar.get(), "fold", L, nL, fold_pct);
      const std::string p = opt.layer_prefix + std::to_string(L) + ".";
      auto qb = [&](const std::string& leaf) { return tbits(p + leaf + ".weight"); };

      // MoE layer? Its MLP is the 3D expert stack (no mlp.gate_proj), so the
      // standard-MLP AWQ fold is skipped and the experts/shared/router are
      // plain-quantized in the main pass. The attention fold below still runs
      // (full-attn or GDN in-projection group), so awq=true does not crash on
      // a MoE checkpoint.
      const bool is_moe_layer =
          src->info(p + "mlp.experts.gate_up_proj") != nullptr;

      // ---- MLP (dense layers only; identical for full-attn and GDN) ----
      int Ng = 0, Kg = 0, Nu = 0, Ku = 0, Nd = 0, Kd = 0, Hi, Hp;
      std::vector<float> Wg, Wu, Wd, sg, sd;
      auto inln = load_f32(p + "input_layernorm.weight", Hi, Hp);
      auto postln = load_f32(p + "post_attention_layernorm.weight", Hp, Hi);
      if (inln.empty() || postln.empty()) {
        return fail("smoothquant: norm load failed at layer " +
                    std::to_string(L));
      }
      if (!is_moe_layer) {
        Wg = load_f32(p + "mlp.gate_proj.weight", Ng, Kg);
        Wu = load_f32(p + "mlp.up_proj.weight", Nu, Ku);
        Wd = load_f32(p + "mlp.down_proj.weight", Nd, Kd);
        if (Wg.empty() || Wu.empty() || Wd.empty()) {
          return fail("smoothquant: MLP load failed at layer " +
                      std::to_string(L));
        }
      }
      // Zero-centered RMSNorm fold (raw-HF Qwen3.5/3.6): the deployed norm is
      // (1 + weight), so fold the +1 BEFORE the AWQ scale folds below (which
      // divide the norm by the per-channel scale) and the final norm_write.
      if (opt.norm_offset) {
        for (float& x : inln) { x += 1.0f; }
        for (float& x : postln) { x += 1.0f; }
      }
      if (!is_moe_layer) {
        // s_gateup (shared input = post_attention_layernorm output).
        awq_search(&cg[(std::size_t)L * H], H,
                   {{Wg.data(), Ng}, {Wu.data(), Nu}},
                   opt.bits, opt.group, false, RS, sg);
        for (int c = 0; c < H; ++c) { postln[c] /= sg[c]; }
        scale_cols(Wg, Ng, H, sg, false); scale_cols(Wu, Nu, H, sg, false);
        // s_down (input = silu(gate)*up): fold into up rows, down cols.
        awq_search(&cd[(std::size_t)L * FFN], FFN, {{Wd.data(), Nd}},
                   opt.bits, opt.group, true, RS, sd);
        scale_rows(Wu, Nu, Ku, sd);          // up out-channel c *= sd[c]
        scale_cols(Wd, Nd, FFN, sd, true);   // down in-channel c *= 1/sd[c]
      }
      // MoE shared gate/up input fold: experts SHARE the post_attention_
      // layernorm output, so derive ONE per-input-channel scale s[H] from
      // calib_gateup[L] and fold 1/s into post_attention_layernorm here; the
      // main loop folds s into the input columns of the router, every expert
      // gate/up, the shared expert gate/up, and the shared-expert gate (all
      // identical-input consumers -> Y unchanged). The alpha is picked from a
      // representative aggregate of those identical-input weights (router +
      // shared expert gate/up; cheap, no big expert load) -- the fold itself
      // is fp-exact regardless of the set. moe_s_in[L] feeds the main loop.
      if (is_moe_layer && !moe_cgu.empty()) {
        std::vector<std::vector<float>> hold;   // own the f32 representatives
        hold.reserve(3 + 16);   // router/shared + up to 16 sampled experts;
                                // reserved so rmats data() pointers stay valid
        std::vector<std::pair<const float*, int>> rmats;
        auto add_rep = [&](const char* leaf) {
          int Nr, Kr;
          auto W = load_f32(p + leaf + ".weight", Nr, Kr);
          if (!W.empty() && Kr == H) {
            hold.push_back(std::move(W));
            rmats.emplace_back(hold.back().data(), Nr);
          }
        };
        add_rep("mlp.gate");
        add_rep("mlp.shared_expert.gate_proj");
        add_rep("mlp.shared_expert.up_proj");
        // The fold is APPLIED to all 256 experts' gate/up, so optimize the
        // alpha over the EXPERTS (which dominate), not just the router/shared
        // proxy. Sample a spread of experts from the fused [E,2I,H] slab
        // (rows [0:I]=gate, [I:2I]=up) into f32 and add to the search set.
        if (moe_E > 0 && moe_I > 0) {
          SharedBuffer gu = src->load(p + "mlp.experts.gate_up_proj", _mc);
          if (!gu.empty()) {
            const auto* g16 = static_cast<const std::uint16_t*>(gu.contents());
            const std::size_t twoI = (std::size_t)2 * moe_I;
            const int n_s = std::min(moe_E, 16);
            const int step = std::max(1, moe_E / n_s);
            for (int e = 0; e < moe_E; e += step) {
              std::vector<float> W((std::size_t)twoI * H);  // expert e gate|up
              const std::size_t base = (std::size_t)e * twoI * H;
              for (std::size_t i = 0; i < W.size(); ++i) {
                W[i] = bf16_to_f32_(g16[base + i]);
              }
              hold.push_back(std::move(W));
              rmats.emplace_back(hold.back().data(), (int)twoI);
            }
          }
        }
        if (!rmats.empty()) {
          std::vector<float> sin;
          awq_search(&moe_cgu[(std::size_t)L * H], H, rmats, opt.bits,
                     opt.group, false, RS, sin);
          for (int c = 0; c < H; ++c) { postln[c] /= sin[c]; }
          moe_s_in[L] = std::move(sin);
        }
      }

      // ---- attention: full-attn (q/k/v/o) OR gated-DeltaNet (in_proj_*) ----
      // The "in-projection group" shares the input_layernorm output (K==H);
      // a single AWQ scale s folds 1/s into input_layernorm and s into every
      // group member's columns (fp-equivalent: the attn/GDN core reads an
      // unchanged W@x). The output projection (o_proj / out_proj) reads a
      // post-core activation with no foldable preceding linear, so -- like the
      // full-attn o_proj has always been -- it is quantized but NOT scale-
      // folded (it still takes the optional per-group clip).
      const bool is_gdn =
          src->info(p + "linear_attn.in_proj_qkv.weight") != nullptr;
      std::vector<Mat> ingroup;   // share input_layernorm output (K==H)
      Mat outproj;                // o_proj / out_proj (quantized, no fold)
      auto load_into = [&](std::vector<Mat>& dst, const std::string& leaf)
          -> bool {
        Mat mat; mat.name = leaf;
        mat.W = load_f32(p + leaf + ".weight", mat.N, mat.K);
        if (mat.W.empty()) { return false; }
        dst.push_back(std::move(mat));
        return true;
      };
      bool lok = true;
      if (is_gdn) {
        for (const char* leaf : {"linear_attn.in_proj_qkv",
             "linear_attn.in_proj_a", "linear_attn.in_proj_b",
             "linear_attn.in_proj_z"}) {
          lok = lok && load_into(ingroup, leaf);
        }
        outproj.name = "linear_attn.out_proj";
        outproj.W = load_f32(p + "linear_attn.out_proj.weight",
                             outproj.N, outproj.K);
        lok = lok && !outproj.W.empty();
      } else {
        for (const char* leaf : {"self_attn.q_proj", "self_attn.k_proj",
             "self_attn.v_proj"}) {
          lok = lok && load_into(ingroup, leaf);
        }
        outproj.name = "self_attn.o_proj";
        outproj.W = load_f32(p + "self_attn.o_proj.weight",
                             outproj.N, outproj.K);
        lok = lok && !outproj.W.empty();
      }
      if (!lok) {
        return fail("smoothquant: attn load failed at layer " +
                    std::to_string(L));
      }
      // s_in over the input_layernorm-output channels (the "qkv" calib slot,
      // collected for every layer regardless of attention type).
      std::vector<std::pair<const float*, int>> imats;
      for (const auto& mm : ingroup) {
        if (mm.K != H) {
          return fail("smoothquant: in-proj K!=H at layer " +
                      std::to_string(L) + " (" + mm.name + ")");
        }
        imats.emplace_back(mm.W.data(), mm.N);
      }
      std::vector<float> si;
      awq_search(&cq[(std::size_t)L * H], H, imats, opt.bits, opt.group, false,
                 RS, si);
      for (int c = 0; c < H; ++c) { inln[c] /= si[c]; }
      for (auto& mm : ingroup) { scale_cols(mm.W, mm.N, H, si, false); }

      // ---- optional AWQ per-group clip (on the folded weights) ----
      if (opt.awq_clip) {
        std::vector<float> ain((std::size_t)H);
        for (int c = 0; c < H; ++c) {
          ain[(std::size_t)c] = cq[(std::size_t)L * H + c] / si[c];
        }
        for (auto& mm : ingroup) {
          awq_clip_search(mm.W.data(), mm.N, H, qb(mm.name), opt.group,
                          ain.data());
        }
        // o_proj clip uses its own attn-output calib (full-attn only).
        if (!is_gdn && QD > 0 && !co.empty()) {
          awq_clip_search(outproj.W.data(), outproj.N, outproj.K,
                          qb(outproj.name), opt.group,
                          &co[(std::size_t)L * QD]);
        }
        if (!is_moe_layer) {
          std::vector<float> agu((std::size_t)H);
          for (int c = 0; c < H; ++c) {
            agu[(std::size_t)c] = cg[(std::size_t)L * H + c] / sg[c];
          }
          std::vector<float> ad((std::size_t)FFN);
          for (int c = 0; c < FFN; ++c) {
            ad[(std::size_t)c] = cd[(std::size_t)L * FFN + c] * sd[c];
          }
          awq_clip_search(Wg.data(), Ng, H, qb("mlp.gate_proj"), opt.group,
                          agu.data());
          awq_clip_search(Wu.data(), Nu, H, qb("mlp.up_proj"), opt.group,
                          agu.data());
          awq_clip_search(Wd.data(), Nd, FFN, qb("mlp.down_proj"), opt.group,
                          ad.data());
        }
      }

      // ---- write ----
      bool ok = true;
      for (const auto& mm : ingroup) {
        ok = ok && quant_write(p + mm.name, mm.W, mm.N, mm.K, qb(mm.name));
      }
      ok = ok &&
          quant_write(p + outproj.name, outproj.W, outproj.N, outproj.K,
                      qb(outproj.name)) &&
          norm_write(p + "input_layernorm.weight", inln) &&
          norm_write(p + "post_attention_layernorm.weight", postln);
      // Dense MLP (skipped on MoE layers -- experts plain-quantized later).
      if (ok && !is_moe_layer) {
        ok = quant_write(p + "mlp.gate_proj", Wg, Ng, Kg,
                         qb("mlp.gate_proj")) &&
             quant_write(p + "mlp.up_proj", Wu, Nu, Ku, qb("mlp.up_proj")) &&
             quant_write(p + "mlp.down_proj", Wd, Nd, Kd, qb("mlp.down_proj"));
      }
      if (!ok) {
        return fail("smoothquant: write failed at layer " + std::to_string(L));
      }
      handled.insert(p + "input_layernorm.weight");
      handled.insert(p + "post_attention_layernorm.weight");
      handled.insert(p + outproj.name + ".weight");
      for (const auto& mm : ingroup) { handled.insert(p + mm.name + ".weight"); }
      if (!is_moe_layer) {
        for (const char* sfx : {"mlp.gate_proj.weight", "mlp.up_proj.weight",
                                "mlp.down_proj.weight"}) {
          handled.insert(p + sfx);
        }
      }
      // ingroup + out (+ gate/up/down on dense layers).
      n_quant += (int)ingroup.size() + 1 + (is_moe_layer ? 0 : 3);
    }
  }

  // Quantize a 2D fp linear [N,K] (bf16/f16 source buffer) at `bits`, with an
  // OPTIONAL per-input-channel column scale `col_scale[K]` (the MoE shared
  // gate/up input fold: W[:,c] *= col_scale[c], fp-equivalent to the 1/scale
  // folded into post_attention_layernorm). Writes the affine triple under pfx.
  auto quant_2d_buf = [&](const std::string& pfx, const SharedBuffer& in,
                          const std::string& dtype, int N, int K, int bits,
                          const float* col_scale) -> bool {
    SharedBuffer w, s, b;
    bool ok;
    if (col_scale != nullptr) {
      SharedBuffer f = _mc->make_shared_buffer((std::size_t)N * K * 2);
      if (f.empty()) { return false; }
      auto* d = static_cast<_Float16*>(f.contents());
      if (dtype == "BF16") {
        const auto* sp = static_cast<const std::uint16_t*>(in.contents());
        for (int n = 0; n < N; ++n) {
          for (int c = 0; c < K; ++c) {
            d[(std::size_t)n * K + c] =
                (_Float16)(bf16_to_f32_(sp[(std::size_t)n * K + c]) *
                           col_scale[c]);
          }
        }
      } else {
        const auto* sp = static_cast<const _Float16*>(in.contents());
        for (int n = 0; n < N; ++n) {
          for (int c = 0; c < K; ++c) {
            d[(std::size_t)n * K + c] =
                (_Float16)((float)sp[(std::size_t)n * K + c] * col_scale[c]);
          }
        }
      }
      ok = q.quantize_linear(f, false, N, K, bits, opt.group, opt.clip,
                             w, s, b);
    } else {
      ok = q.quantize_linear(in, dtype == "BF16", N, K, bits, opt.group,
                             opt.clip, w, s, b);
    }
    if (!ok) { return false; }
    const std::int64_t wcols = (std::int64_t)K * bits / 32;
    const std::int64_t gcols = (std::int64_t)K / opt.group;
    return wr.add(pfx + ".weight", "U32", {N, wcols}, w.contents(),
                  w.byte_size()) &&
           wr.add(pfx + ".scales", "F16", {N, gcols}, s.contents(),
                  s.byte_size()) &&
           wr.add(pfx + ".biases", "F16", {N, gcols}, b.contents(),
                  b.byte_size());
  };

  // ---- MoE 3D expert bridge (raw-HF Qwen3.5/3.6 MoE) ----------------------
  // The HF checkpoint stores experts as fused 3D slabs; vpipe's MoE loader
  // wants per-projection switch_mlp.* affine triples. Quantize each expert's
  // [rows, K] slab (group-affine, opt.bits) and assemble them contiguously
  // into one 3D tensor [E, rows, ...] -- exactly the layout interleave_moe
  // reads straight through. `src3d` is bf16; per expert e the slab starts at
  // element (e*estride + roff) and is `rows` rows of K columns.
  auto quant_experts = [&](const std::string& pfx, const std::uint16_t* src3d,
                           int E, int rows, int K, std::size_t estride,
                           std::size_t roff, int bits) -> bool {
    const std::int64_t wcols = (std::int64_t)K * bits / 32;
    const std::int64_t gcols = (std::int64_t)K / opt.group;
    const std::size_t we = (std::size_t)rows * (std::size_t)wcols * 4;
    const std::size_t ge = (std::size_t)rows * (std::size_t)gcols * 2;
    SharedBuffer W = _mc->make_shared_buffer((std::size_t)E * we);
    SharedBuffer S = _mc->make_shared_buffer((std::size_t)E * ge);
    SharedBuffer B = _mc->make_shared_buffer((std::size_t)E * ge);
    SharedBuffer slab = _mc->make_shared_buffer((std::size_t)rows * K * 2);
    if (W.empty() || S.empty() || B.empty() || slab.empty()) { return false; }
    auto* sd = static_cast<std::uint16_t*>(slab.contents());
    for (int e = 0; e < E; ++e) {
      std::memcpy(sd, src3d + (std::size_t)e * estride + roff,
                  (std::size_t)rows * K * 2);
      SharedBuffer w, s, b;
      if (!q.quantize_linear(slab, /*src_bf16=*/true, rows, K, bits, opt.group,
                             opt.clip, w, s, b)) {
        return false;
      }
      std::memcpy((char*)W.contents() + (std::size_t)e * we,
                  w.contents(), we);
      std::memcpy((char*)S.contents() + (std::size_t)e * ge,
                  s.contents(), ge);
      std::memcpy((char*)B.contents() + (std::size_t)e * ge,
                  b.contents(), ge);
    }
    return wr.add(pfx + ".weight", "U32", {E, rows, wcols}, W.contents(),
                  W.byte_size()) &&
           wr.add(pfx + ".scales", "F16", {E, rows, gcols}, S.contents(),
                  S.byte_size()) &&
           wr.add(pfx + ".biases", "F16", {E, rows, gcols}, B.contents(),
                  B.byte_size());
  };
  // As quant_experts, but each expert's f32 [rows,K] slab is produced by a
  // `fold` callback (applies the per-expert AWQ folds/clip in host f32) before
  // quantizing. One expert's f32 is held at a time (~rows*K*4 bytes); the bf16
  // source stays mmapped/loaded by the caller. fp-equivalent: the folds cancel
  // against the matching norm/up-row fold, so the MoE output is unchanged.
  auto quant_experts_fold =
      [&](const std::string& pfx, const std::uint16_t* src3d, int E, int rows,
          int K, std::size_t estride, std::size_t roff, int bits,
          const std::function<void(int, const std::uint16_t*, float*)>& fold)
      -> bool {
    const std::int64_t wcols = (std::int64_t)K * bits / 32;
    const std::int64_t gcols = (std::int64_t)K / opt.group;
    const std::size_t we = (std::size_t)rows * (std::size_t)wcols * 4;
    const std::size_t ge = (std::size_t)rows * (std::size_t)gcols * 2;
    SharedBuffer W = _mc->make_shared_buffer((std::size_t)E * we);
    SharedBuffer S = _mc->make_shared_buffer((std::size_t)E * ge);
    SharedBuffer B = _mc->make_shared_buffer((std::size_t)E * ge);
    SharedBuffer slab = _mc->make_shared_buffer((std::size_t)rows * K * 2);
    if (W.empty() || S.empty() || B.empty() || slab.empty()) { return false; }
    auto* sd = static_cast<_Float16*>(slab.contents());
    std::vector<float> tmp((std::size_t)rows * K);
    for (int e = 0; e < E; ++e) {
      fold(e, src3d + (std::size_t)e * estride + roff, tmp.data());
      for (std::size_t i = 0; i < tmp.size(); ++i) { sd[i] = (_Float16)tmp[i]; }
      SharedBuffer w, s, b;
      if (!q.quantize_linear(slab, /*src_bf16=*/false, rows, K, bits,
                             opt.group, opt.clip, w, s, b)) {
        return false;
      }
      std::memcpy((char*)W.contents() + (std::size_t)e * we,
                  w.contents(), we);
      std::memcpy((char*)S.contents() + (std::size_t)e * ge,
                  s.contents(), ge);
      std::memcpy((char*)B.contents() + (std::size_t)e * ge,
                  b.contents(), ge);
    }
    return wr.add(pfx + ".weight", "U32", {E, rows, wcols}, W.contents(),
                  W.byte_size()) &&
           wr.add(pfx + ".scales", "F16", {E, rows, gcols}, S.contents(),
                  S.byte_size()) &&
           wr.add(pfx + ".biases", "F16", {E, rows, gcols}, B.contents(),
                  B.byte_size());
  };
  // Quantize a 2D fp linear [N,K] at a forced bit width (router mlp.gate +
  // shared_expert_gate are w8 by the MoE convention, independent of opt.bits),
  // with an optional MoE shared gate/up input col fold (col_scale[K]).
  auto quant_2d_at = [&](const std::string& name, int bits,
                         const float* col_scale) -> bool {
    const auto* ti = src->info(name);
    if (ti == nullptr || ti->shape.size() != 2) { return false; }
    const int N = (int)ti->shape[0], K = (int)ti->shape[1];
    if (K % opt.group != 0) { return false; }
    SharedBuffer in = src->load(name, _mc);
    if (in.empty()) { return false; }
    const std::string pfx = name.substr(0, name.size() - 7);   // ".weight"
    return quant_2d_buf(pfx, in, ti->dtype, N, K, bits, col_scale);
  };

  int quant_pct = -1;
  int idx = 0;
  for (const auto& name : names) {
    if (stop()) { return fail("model-quantize: stopped by request"); }
    ++idx;
    quant_progress_(bar.get(), "quantize", idx, (int)names.size(),
                    quant_pct);
    if (handled.count(name) > 0) { continue; }
    const auto* ti = src->info(name);
    if (ti == nullptr) { continue; }

    // MTP head: the in-shard affine MTP loader decodes it as 4-bit g64, so
    // only emit the mtp.* tensors at that setting; otherwise drop them (the
    // head is then simply absent -> has_mtp() false -> pdecode fallback)
    // rather than writing a head the loader would mis-decode. At 4-bit g64
    // they flow through the generic path below: linears quantize by leaf
    // match, fc passes through, and the norms stay RAW (excluded from the
    // +1 fold) so the loader's load-time +1 is correct.
    if ((name.rfind("mtp.", 0) == 0 ||
         name.find(".mtp.") != std::string::npos) &&
        (opt.bits != 4 || opt.group != 64)) {
      continue;
    }

    // ---- MoE expert slabs (3D) -> switch_mlp.* affine triples -----------
    if (ends_with(name, ".mlp.experts.gate_up_proj")) {
      if (ti->shape.size() != 3 || ti->dtype != "BF16") {
        return fail("model-quantize: bad experts.gate_up_proj: " + name);
      }
      const int E = (int)ti->shape[0], twoI = (int)ti->shape[1];
      const int H = (int)ti->shape[2], I = twoI / 2;
      if ((twoI & 1) || (H % opt.group) != 0) {
        return fail("model-quantize: bad gate_up dims: " + name);
      }
      SharedBuffer in = src->load(name, _mc);
      if (in.empty()) {
        return fail("model-quantize: load failed: " + name);
      }
      const auto* sp = static_cast<const std::uint16_t*>(in.contents());
      // base = "...mlp" (strip ".experts.gate_up_proj").
      const std::string base =
          name.substr(0, name.size() - 21);
      const std::size_t estride = (std::size_t)twoI * H;
      // HF concatenates gate (rows [0:I]) then up (rows [I:2I]) per slab.
      const int L = layer_of(name);
      const bool fold = L >= 0 && moe_s_in.count(L) > 0;
      if (fold) {
        // Shared gate/up input fold s[H] (cols) + per-expert gate/up clip
        // (calib_expert_gateup, act/s) + per-expert down fold s_e[I] on the
        // up OUTPUT rows (paired with the 1/s_e in down cols below).
        const std::vector<float>& sIn = moe_s_in[L];          // [H]
        const std::vector<float>& sDn = moe_s_down[L];        // [E*I]
        const float* ceg = moe_ceg.data() +
                           (std::size_t)L * moe_E * moe_H;     // [E*H]
        auto clip_gu = [&](int e, float* dst) {
          if (!opt.awq_clip) { return; }
          const float* eg = ceg + (std::size_t)e * H;
          bool zero = true;
          for (int c = 0; c < H; ++c) { if (eg[c] != 0.0f) { zero = false;
                                                              break; } }
          if (zero) { return; }
          std::vector<float> a((std::size_t)H);
          for (int c = 0; c < H; ++c) { a[(std::size_t)c] = eg[c] / sIn[c]; }
          awq_clip_search(dst, I, H, opt.bits, opt.group, a.data());
        };
        auto fold_gate = [&](int e, const std::uint16_t* se, float* dst) {
          for (int n = 0; n < I; ++n) {
            for (int c = 0; c < H; ++c) {
              dst[(std::size_t)n * H + c] =
                  bf16_to_f32_(se[(std::size_t)n * H + c]) * sIn[c];
            }
          }
          clip_gu(e, dst);
        };
        auto fold_up = [&](int e, const std::uint16_t* se, float* dst) {
          const float* sr = sDn.data() + (std::size_t)e * I;   // [I]
          for (int n = 0; n < I; ++n) {
            const float rn = sr[n];
            for (int c = 0; c < H; ++c) {
              dst[(std::size_t)n * H + c] =
                  bf16_to_f32_(se[(std::size_t)n * H + c]) * sIn[c] * rn;
            }
          }
          clip_gu(e, dst);
        };
        if (!quant_experts_fold(base + ".switch_mlp.gate_proj", sp, E, I, H,
                                estride, 0, opt.bits, fold_gate) ||
            !quant_experts_fold(base + ".switch_mlp.up_proj", sp, E, I, H,
                                estride, (std::size_t)I * H, opt.bits,
                                fold_up)) {
          return fail("model-quantize: expert gate/up fold failed: " + name);
        }
      } else if (!quant_experts(base + ".switch_mlp.gate_proj", sp, E, I, H,
                                estride, 0, opt.bits) ||
                 !quant_experts(base + ".switch_mlp.up_proj", sp, E, I, H,
                                estride, (std::size_t)I * H, opt.bits)) {
        return fail("model-quantize: expert gate/up quant failed: " + name);
      }
      n_quant += 2;
      continue;
    }
    if (ends_with(name, ".mlp.experts.down_proj")) {
      if (ti->shape.size() != 3 || ti->dtype != "BF16") {
        return fail("model-quantize: bad experts.down_proj: " + name);
      }
      const int E = (int)ti->shape[0], H = (int)ti->shape[1];
      const int I = (int)ti->shape[2];
      if ((I % opt.group) != 0) {
        return fail("model-quantize: bad down dims: " + name);
      }
      SharedBuffer in = src->load(name, _mc);
      if (in.empty()) {
        return fail("model-quantize: load failed: " + name);
      }
      const auto* sp = static_cast<const std::uint16_t*>(in.contents());
      // base = "...mlp" (strip ".experts.down_proj" = 18 chars).
      const std::string base = name.substr(0, name.size() - 18);
      const int L = layer_of(name);
      const bool fold = L >= 0 && moe_s_in.count(L) > 0;
      if (fold) {
        // Per-expert down input fold: derive s_e[I] from calib_expert_down
        // (the silu(gate_e)*up_e magnitude; AWQ inv form) and fold 1/s_e into
        // the down INPUT columns here -- the matching s_e on the up OUTPUT
        // rows is applied in the gate_up branch (processed AFTER down in the
        // sorted name order). Zero-stat (unrouted) experts -> s_e = 1 (skip).
        const float* ced = moe_ced.data() +
                           (std::size_t)L * moe_E * moe_I;     // [E*I]
        std::vector<float> sdn((std::size_t)E * I, 1.0f);
        std::vector<float> de((std::size_t)H * I);             // one [H,I]
        const int RS = 8;
        for (int e = 0; e < E; ++e) {
          const float* ce = ced + (std::size_t)e * I;
          bool zero = true;
          for (int c = 0; c < I; ++c) { if (ce[c] != 0.0f) { zero = false;
                                                              break; } }
          if (zero) { continue; }
          const std::uint16_t* se = sp + (std::size_t)e * H * I;
          for (std::size_t i = 0; i < de.size(); ++i) {
            de[i] = bf16_to_f32_(se[i]);
          }
          std::vector<float> s_e;
          awq_search(ce, I, {{de.data(), H}}, opt.bits, opt.group, true, RS,
                     s_e);
          for (int c = 0; c < I; ++c) { sdn[(std::size_t)e * I + c] = s_e[c]; }
        }
        moe_s_down[L] = std::move(sdn);
        const std::vector<float>& sDn = moe_s_down[L];
        auto fold_down = [&](int e, const std::uint16_t* se, float* dst) {
          const float* s_e = sDn.data() + (std::size_t)e * I;  // [I]
          for (int n = 0; n < H; ++n) {
            for (int c = 0; c < I; ++c) {
              dst[(std::size_t)n * I + c] =
                  bf16_to_f32_(se[(std::size_t)n * I + c]) / s_e[c];
            }
          }
        };
        if (!quant_experts_fold(base + ".switch_mlp.down_proj", sp, E, H, I,
                                (std::size_t)H * I, 0, opt.bits, fold_down)) {
          return fail("model-quantize: expert down fold failed: " + name);
        }
      } else if (!quant_experts(base + ".switch_mlp.down_proj", sp, E, H, I,
                                (std::size_t)H * I, 0, opt.bits)) {
        return fail("model-quantize: expert down quant failed: " + name);
      }
      if (S) { S->log_debug(fmt("  experts {}", name)); }
      ++n_quant;
      continue;
    }
    // ---- Gemma-4 MoE expert slabs (3D) -> per-projection affine triples ----
    // Gemma names the fused slabs `<layer>.experts.gate_up_proj` [E,2I,H]
    // (gate rows [0:I], up rows [I:2I]) and `<layer>.experts.down_proj`
    // [E,H,I], WITHOUT the Qwen `.mlp.` segment (so those branches above skip
    // them). Quantize per-expert group-affine at opt.bits and write gemma-
    // natural `<layer>.experts.{gate,up,down}_proj.{weight,scales,biases}`
    // triples the MetalGemmaModel MoE loader reads. Placed AFTER the Qwen
    // branches so a Qwen `.mlp.experts.*` name is consumed there first.
    if (ends_with(name, ".experts.gate_up_proj")) {
      if (ti->shape.size() != 3 || ti->dtype != "BF16") {
        return fail("model-quantize: bad experts.gate_up_proj: " + name);
      }
      const int E = (int)ti->shape[0], twoI = (int)ti->shape[1];
      const int H = (int)ti->shape[2], I = twoI / 2;
      if ((twoI & 1) || (H % opt.group) != 0) {
        return fail("model-quantize: bad gate_up dims: " + name);
      }
      SharedBuffer in = src->load(name, _mc);
      if (in.empty()) {
        return fail("model-quantize: load failed: " + name);
      }
      const auto* sp = static_cast<const std::uint16_t*>(in.contents());
      const std::string base = name.substr(0, name.size() - 21);
      const std::size_t estride = (std::size_t)twoI * H;
      if (!quant_experts(base + ".experts.gate_proj", sp, E, I, H, estride, 0,
                         opt.bits) ||
          !quant_experts(base + ".experts.up_proj", sp, E, I, H, estride,
                         (std::size_t)I * H, opt.bits)) {
        return fail("model-quantize: gemma expert gate/up quant: " + name);
      }
      if (S) { S->log_debug(fmt("  experts {}", name)); }
      n_quant += 2;
      continue;
    }
    if (ends_with(name, ".experts.down_proj")) {
      if (ti->shape.size() != 3 || ti->dtype != "BF16") {
        return fail("model-quantize: bad experts.down_proj: " + name);
      }
      const int E = (int)ti->shape[0], H = (int)ti->shape[1];
      const int I = (int)ti->shape[2];
      if ((I % opt.group) != 0) {
        return fail("model-quantize: bad down dims: " + name);
      }
      SharedBuffer in = src->load(name, _mc);
      if (in.empty()) {
        return fail("model-quantize: load failed: " + name);
      }
      const auto* sp = static_cast<const std::uint16_t*>(in.contents());
      const std::string base = name.substr(0, name.size() - 18);
      if (!quant_experts(base + ".experts.down_proj", sp, E, H, I,
                         (std::size_t)H * I, 0, opt.bits)) {
        return fail("model-quantize: gemma expert down quant: " + name);
      }
      if (S) { S->log_debug(fmt("  experts {}", name)); }
      ++n_quant;
      continue;
    }
    // Router + shared-expert gate -> w8 (the MoE w8 convention), regardless of
    // opt.bits, so the loader's qtri reads them as 8-bit affine. Both read the
    // shared post_attention_layernorm output, so they take the same s[H] input
    // col fold as the experts (fp-equivalent; K==H guard).
    if (ends_with(name, ".mlp.gate.weight") ||
        ends_with(name, ".mlp.shared_expert_gate.weight")) {
      const int L = layer_of(name);
      const float* cs = (L >= 0 && moe_s_in.count(L) > 0 &&
                         ti->shape.size() == 2 && (int)ti->shape[1] == moe_H)
                            ? moe_s_in[L].data()
                            : nullptr;
      if (!quant_2d_at(name, 8, cs)) {
        return fail("model-quantize: w8 router/gate quant failed: " + name);
      }
      if (S) { S->log_debug(fmt("  router {}", name)); }
      ++n_quant;
      continue;
    }

    // Zero-centered RMSNorm fold: rewrite weight -> 1 + weight (bf16).
    if (norm_plus_one(name)) {
      int Nn, Kn;
      std::vector<float> v = load_f32(name, Nn, Kn);
      if (v.empty()) { return fail("model-quantize: norm load failed: " + name); }
      for (float& x : v) { x += 1.0f; }
      if (!norm_write(name, v)) {
        return fail("model-quantize: norm write failed: " + name);
      }
      ++n_pass;
      continue;
    }

    const std::string leaf = weight_leaf_(name);
    const bool is_2d = ti->shape.size() == 2;
    const bool fp = ti->dtype == "BF16" || ti->dtype == "F16";
    const bool quant = !leaf.empty() && quant_set.count(leaf) > 0 && is_2d &&
                       fp && (ti->shape[1] % opt.group == 0);

    SharedBuffer in = src->load(name, _mc);
    if (in.empty()) { return fail("model-quantize: load failed: " + name); }

    if (!quant) {
      if (!wr.add(name, ti->dtype, ti->shape, in.contents(), in.byte_size())) {
        return fail("model-quantize: write (passthrough) failed: " + name);
      }
      if (S) { S->log_debug(fmt("  pass {}", name)); }
      ++n_pass;
      continue;
    }

    const int N = (int)ti->shape[0], K = (int)ti->shape[1];
    const int tb = tbits(name);
    // MoE shared-expert gate/up read the shared post_attention_layernorm
    // output -> same s[H] input col fold as the experts (down NOT folded; its
    // input is the post-activation, not the shared norm output).
    const float* gcs = nullptr;
    if (!moe_s_in.empty() &&
        (ends_with(name, ".mlp.shared_expert.gate_proj.weight") ||
         ends_with(name, ".mlp.shared_expert.up_proj.weight"))) {
      const int L = layer_of(name);
      if (L >= 0 && moe_s_in.count(L) > 0 && K == moe_H) {
        gcs = moe_s_in[L].data();
      }
    }
    const std::string pfx = name.substr(0, name.size() - 7);  // ".weight"
    if (S) {
      S->log_debug(fmt("  quant {} [{}x{}] {}-bit", name, N, K, tb));
    }
    if (!quant_2d_buf(pfx, in, ti->dtype, N, K, tb, gcs)) {
      return fail("model-quantize: write (quant) failed: " + name);
    }
    ++n_quant;
  }
  if (bar) { bar->end(); }   // finalize the bar line before the summary

  if (!wr.close()) { return fail("model-quantize: finalize shards failed"); }

  // Rewrite config.json with the top-level quantization block (the loader
  // reads outer.quantization.{bits,group_size}); copy everything else.
  {
    std::string cfg_txt;
    const std::string cfg_in = (fs::path(in_dir) / "config.json").string();
    if (!read_file_(cfg_in, &cfg_txt)) {
      return fail("model-quantize: cannot read config.json");
    }
    FlexData cfg;
    try { cfg = FlexData::from_json(cfg_txt); }
    catch (...) { return fail("model-quantize: bad config.json"); }
    if (!cfg.is_object()) { return fail("model-quantize: config not object"); }
    FlexData qb = FlexData::make_object();
    {
      auto o = qb.as_object();
      o.insert_or_assign("group_size", FlexData::make_int(opt.group));
      o.insert_or_assign("bits", FlexData::make_int(opt.bits));
    }
    cfg.as_object().insert_or_assign("quantization", std::move(qb));
    if (!write_file_((fs::path(out_dir) / "config.json").string(),
                     cfg.to_json(true))) {
      return fail("model-quantize: cannot write config.json");
    }
  }

  // Copy sidecar files (tokenizer, processor, etc.) -- everything that is
  // not a weight file or config.json (already rewritten).
  for (const auto& de : fs::directory_iterator(in_dir, ec)) {
    if (!de.is_regular_file()) { continue; }
    const std::string fn = de.path().filename().string();
    if (fn == "config.json" || fn.find(".safetensors") != std::string::npos) {
      continue;
    }
    fs::copy_file(de.path(), fs::path(out_dir) / fn,
                  fs::copy_options::overwrite_existing, ec);
  }

  if (_mc->session() != nullptr) {
    _mc->session()->info(fmt(
        "model-quantize: {} -> {} ({}-bit g{}): {} quantized, {} passthrough",
        in_dir, out_dir, opt.bits, opt.group, n_quant, n_pass));
  }
  return true;
}

}  // namespace vpipe::genai
