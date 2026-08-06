#include "generative-models/wan/metal-umt5-encoder.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

std::string
blk_(int i, const char* rest)
{
  return "encoder.block." + std::to_string(i) + "." + rest;
}

}  // namespace

bool
MetalUmt5Encoder::config_from_json(const std::string& enc_dir, Config& out,
                                   std::string* err)
{
  namespace fs = std::filesystem;
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  fs::path p(enc_dir);
  if (fs::is_directory(p) && !fs::exists(p / "config.json")) {
    p = p / "text_encoder";
  }
  if (fs::is_directory(p)) { p = p / "config.json"; }
  std::ifstream f(p);
  if (!f) { return fail("cannot open " + p.string()); }
  FlexData cfg;
  try {
    cfg = FlexData::from_json(f);
  } catch (...) {
    return fail("cannot parse " + p.string());
  }
  if (!cfg.is_object()) { return fail(p.string() + " is not a JSON object"); }
  auto o = cfg.as_object();
  const std::string mt =
      o.contains("model_type") ? std::string(o.at("model_type").as_string(""))
                               : std::string();
  if (mt != "umt5" && mt != "t5") {
    return fail("not a umt5 encoder config (model_type=" + mt + ")");
  }
  auto gi = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  };
  out.d_model  = gi("d_model", 4096);
  out.d_ff     = gi("d_ff", 10240);
  out.n_heads  = gi("num_heads", 64);
  out.d_kv     = gi("d_kv", 64);
  out.n_layers = gi("num_layers", 24);
  out.vocab    = gi("vocab_size", 256384);
  out.rel_buckets  = gi("relative_attention_num_buckets", 32);
  out.rel_max_dist = gi("relative_attention_max_distance", 128);
  if (o.contains("layer_norm_epsilon")) {
    out.norm_eps = (float)o.at("layer_norm_epsilon").as_real(1e-6);
  }
  // The gated form is what the FF here assumes; a non-gated T5 would need
  // a different feed-forward, so refuse rather than silently misread wi_0.
  if (o.contains("is_gated_act") && !o.at("is_gated_act").as_bool(true)) {
    return fail("non-gated T5 feed-forward is not supported here");
  }
  return true;
}

SharedBuffer
MetalUmt5Encoder::weight_(WeightSet& ws, const std::string& nm)
{
  // The checkpoint is bf16 and so is the forward, so every weight is kept
  // exactly as it lies on disk -- no transform, hence tensor() (cached and
  // PARKABLE) rather than derived(). This is a model that KEEPS all of its
  // ~11 GB, which is precisely what the cache is for.
  return ws.tensor(nm, _mc, WeightSet::Residency::Copied);
}

std::unique_ptr<MetalUmt5Encoder>
MetalUmt5Encoder::load(const std::string& model_dir, MetalCompute* mc,
                       const Config& cfg)
{
  namespace fs = std::filesystem;
  fs::path p(model_dir);
  if (fs::is_directory(p) && fs::exists(p / "text_encoder")) {
    p = p / "text_encoder";
  }
  return load(WeightSet::open(p.string(), nullptr), mc, cfg);
}

std::unique_ptr<MetalUmt5Encoder>
MetalUmt5Encoder::load(std::shared_ptr<WeightSet> ws_in, MetalCompute* mc,
                       const Config& cfg)
{
  if (mc == nullptr || !ws_in) { return nullptr; }
  WeightSet& ws = *ws_in;
  auto m = std::unique_ptr<MetalUmt5Encoder>(new MetalUmt5Encoder());
  m->_ws = std::move(ws_in);
  m->_mc = mc;
  m->_cfg = cfg;

  // BF16 metallibs: see the header for why this tower is not f16.
  m->_lib_gemm = mc->load_library("dense_gemm_bf16");
  m->_lib_elt  = mc->load_library("llm_elementwise_bf16");
  m->_lib_rms  = mc->load_library("rms_norm_bf16");
  m->_fn_gemm     = m->_lib_gemm.function("dense_gemm_bias_f16");
  m->_fn_rms      = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_geglu    = m->_lib_elt.function("geglu_f16");
  m->_fn_residual = m->_lib_elt.function("residual_add_f16");
  m->_fn_attn     = m->_lib_elt.function("umt5_attn_f16");
  if (!m->_fn_gemm.valid() || !m->_fn_rms.valid() || !m->_fn_geglu.valid() ||
      !m->_fn_residual.valid() || !m->_fn_attn.valid()) {
    return nullptr;
  }

  m->_embed = m->weight_(ws, "shared.weight");
  m->_final_norm = m->weight_(ws, "encoder.final_layer_norm.weight");
  if (m->_embed.empty() || m->_final_norm.empty()) { return nullptr; }

  m->_layers.resize((std::size_t)cfg.n_layers);
  for (int i = 0; i < cfg.n_layers; ++i) {
    Layer& l = m->_layers[(std::size_t)i];
    l.n1  = m->weight_(ws, blk_(i, "layer.0.layer_norm.weight"));
    l.q   = m->weight_(ws, blk_(i, "layer.0.SelfAttention.q.weight"));
    l.k   = m->weight_(ws, blk_(i, "layer.0.SelfAttention.k.weight"));
    l.v   = m->weight_(ws, blk_(i, "layer.0.SelfAttention.v.weight"));
    l.o   = m->weight_(ws, blk_(i, "layer.0.SelfAttention.o.weight"));
    // umT5 carries its own bias table in EVERY layer; T5 shares layer 0's.
    l.rel = m->weight_(
        ws, blk_(i, "layer.0.SelfAttention.relative_attention_bias.weight"));
    l.n2  = m->weight_(ws, blk_(i, "layer.1.layer_norm.weight"));
    l.wi0 = m->weight_(ws, blk_(i, "layer.1.DenseReluDense.wi_0.weight"));
    l.wi1 = m->weight_(ws, blk_(i, "layer.1.DenseReluDense.wi_1.weight"));
    l.wo  = m->weight_(ws, blk_(i, "layer.1.DenseReluDense.wo.weight"));
    if (l.n1.empty() || l.q.empty() || l.k.empty() || l.v.empty() ||
        l.o.empty() || l.rel.empty() || l.n2.empty() || l.wi0.empty() ||
        l.wi1.empty() || l.wo.empty()) {
      return nullptr;
    }
  }
  return m;
}

// The relative-position bucket for every (query, key) pair, in the float
// arithmetic the reference uses -- the log's truncation to long is what
// picks the bucket near a boundary, so reproducing the expression matters
// more than the 256 KB the table costs.
std::vector<std::uint8_t>
MetalUmt5Encoder::bucket_table_(int L) const
{
  std::vector<std::uint8_t> t((std::size_t)L * L, 0);
  // Bidirectional: half the buckets carry the sign, so the magnitude is
  // bucketed over num_buckets/2.
  const int nb = _cfg.rel_buckets / 2;             // 16
  const int max_exact = nb / 2;                    // 8
  const float denom =
      std::log((float)_cfg.rel_max_dist / (float)max_exact);
  for (int i = 0; i < L; ++i) {
    for (int j = 0; j < L; ++j) {
      const int rel = j - i;
      int bucket = (rel > 0) ? nb : 0;
      const int a = rel < 0 ? -rel : rel;
      if (a < max_exact) {
        bucket += a;
      } else {
        const float lr =
            std::log((float)a / (float)max_exact) / denom * (float)(nb - max_exact);
        int big = max_exact + (int)lr;
        if (big > nb - 1) { big = nb - 1; }
        bucket += big;
      }
      t[(std::size_t)i * L + j] = (std::uint8_t)bucket;
    }
  }
  return t;
}

void
MetalUmt5Encoder::gemm_(ComputeEncoder& enc, const SharedBuffer& x,
                        const SharedBuffer& w, const SharedBuffer& y, int M,
                        int N, int K)
{
  enc.set_function(_fn_gemm);
  enc.set_buffer(0, x); enc.set_buffer(1, w);
  enc.set_buffer(2, w);            // bias slot unused (has_bias = 0)
  enc.set_buffer(3, y);
  enc.set_constant(4, M); enc.set_constant(5, N); enc.set_constant(6, K);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((N + 15) / 16) * 16),
                (unsigned)(((M + 15) / 16) * 16), 1}, {16, 16, 1});
}

SharedBuffer
MetalUmt5Encoder::encode(const std::vector<std::int32_t>& ids, int n_valid,
                         std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  const int L = (int)ids.size();
  if (L <= 0) { return fail("empty token sequence"); }
  if (L > kMaxSeq) {
    return fail(fmt("sequence of {} exceeds the {}-token attention limit", L,
                    kMaxSeq)());
  }
  if (n_valid < 0 || n_valid > L) { n_valid = L; }
  const int D = _cfg.d_model, F = _cfg.d_ff;
  const int H = _cfg.n_heads, Dh = _cfg.d_kv;
  if (H * Dh != D) {
    return fail("n_heads * d_kv != d_model (unsupported umT5 shape)");
  }
  MetalCompute* mc = _mc;
  const std::size_t rows = (std::size_t)L;

  auto mk = [&](std::size_t elems) { return mc->make_shared_buffer(elems * 2); };
  SharedBuffer h = mk(rows * D);
  SharedBuffer t1 = mk(rows * D), t2 = mk(rows * D);
  SharedBuffer q = mk(rows * D), k = mk(rows * D), v = mk(rows * D);
  SharedBuffer a = mk(rows * D);
  SharedBuffer g = mk(rows * F), u = mk(rows * F), gu = mk(rows * F);
  if (h.empty() || t1.empty() || t2.empty() || q.empty() || k.empty() ||
      v.empty() || a.empty() || g.empty() || u.empty() || gu.empty()) {
    return fail("activation allocation failed (out of GPU memory)");
  }

  // Embedding lookup on the host: a gather of L rows out of a 2 GB table
  // is 4 MB of memcpy, which is not worth a kernel or the round trip.
  {
    const auto* src = static_cast<const std::uint16_t*>(_embed.contents());
    auto* dst = static_cast<std::uint16_t*>(h.contents());
    for (int i = 0; i < L; ++i) {
      const std::int32_t id = ids[(std::size_t)i];
      if (id < 0 || id >= _cfg.vocab) {
        return fail(fmt("token id {} out of range", (long long)id)());
      }
      std::memcpy(dst + (std::size_t)i * D,
                  src + (std::size_t)id * D, (std::size_t)D * 2);
    }
  }

  const std::vector<std::uint8_t> table = bucket_table_(L);
  SharedBuffer buckets = mc->make_shared_buffer(table.size());
  if (buckets.empty()) { return fail("bucket table allocation failed"); }
  std::memcpy(buckets.contents(), table.data(), table.size());

  CommandStream stream = mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto rms = [&](const SharedBuffer& in, const SharedBuffer& gamma,
                   const SharedBuffer& out) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, in); enc.set_buffer(1, gamma); enc.set_buffer(2, out);
      enc.set_constant(3, D); enc.set_constant(4, _cfg.norm_eps);
      enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
    };
    auto add = [&](const SharedBuffer& x, const SharedBuffer& y,
                   const SharedBuffer& out, std::size_t n) {
      enc.set_function(_fn_residual);
      enc.set_buffer(0, x); enc.set_buffer(1, y); enc.set_buffer(2, out);
      enc.set_constant(3, (int)n);
      enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    };

    for (const Layer& l : _layers) {
      // --- self-attention -------------------------------------------
      rms(h, l.n1, t1);
      gemm_(enc, t1, l.q, q, L, D, D);
      gemm_(enc, t1, l.k, k, L, D, D);
      gemm_(enc, t1, l.v, v, L, D, D);
      // NOTE: no 1/sqrt(d) -- T5 folds the scale into initialization, so
      // the kernel takes the raw dot products.
      enc.set_function(_fn_attn);
      enc.set_buffer(0, q); enc.set_buffer(1, k); enc.set_buffer(2, v);
      enc.set_buffer(3, l.rel); enc.set_buffer(4, buckets);
      enc.set_buffer(5, a);
      enc.set_constant(6, L); enc.set_constant(7, H); enc.set_constant(8, Dh);
      enc.set_constant(9, n_valid);
      enc.dispatch({256, (unsigned)L, (unsigned)H}, {256, 1, 1});
      gemm_(enc, a, l.o, t2, L, D, D);
      add(h, t2, h, rows * D);

      // --- gated feed-forward ---------------------------------------
      rms(h, l.n2, t1);
      gemm_(enc, t1, l.wi0, g, L, F, D);
      gemm_(enc, t1, l.wi1, u, L, F, D);
      enc.set_function(_fn_geglu);
      enc.set_buffer(0, g); enc.set_buffer(1, u); enc.set_buffer(2, gu);
      enc.set_constant(3, (int)(rows * F));
      enc.dispatch({(unsigned)(rows * F), 1, 1}, {256, 1, 1});
      gemm_(enc, gu, l.wo, t2, L, D, F);
      add(h, t2, h, rows * D);
    }
    rms(h, _final_norm, t1);
  }
  std::string gpu_err;
  if (!stream.commit().wait_ok(&gpu_err)) {
    return fail(gpu_err.empty() ? std::string("umT5 encode failed") : gpu_err);
  }

  // Zero the padded tail. The Wan pipeline truncates the encoder output to
  // the real token count and pads back with ZEROS, so what the DiT
  // cross-attends to past n_valid is zero -- not whatever the encoder made
  // of the pad tokens. That is part of the conditioning contract.
  {
    auto* d = static_cast<std::uint16_t*>(t1.contents());
    for (int i = n_valid; i < L; ++i) {
      std::memset(d + (std::size_t)i * D, 0, (std::size_t)D * 2);
    }
  }
  return t1;
}

}  // namespace genai
}  // namespace vpipe
