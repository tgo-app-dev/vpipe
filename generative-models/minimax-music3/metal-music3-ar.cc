#include "generative-models/minimax-music3/metal-music3-ar.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/perf-event.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "generative-models/context-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/quantize/affine-quantizer.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/safetensors-ranges.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace vpipe::genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

constexpr const char* kKey = "music3-ar|";

bool
read_json_(const std::filesystem::path& p, FlexData* out)
{
  std::ifstream in(p);
  if (!in) { return false; }
  try {
    *out = FlexData::from_json(in);
  } catch (const std::exception&) {
    return false;
  }
  return out->is_object();
}

inline float
bf16_f32_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

inline std::uint16_t
f32_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) { return (std::uint16_t)0x7fc0; }
  u += 0x7fffu + ((u >> 16) & 1u);
  return (std::uint16_t)(u >> 16);
}

}  // namespace

// ---- config -----------------------------------------------------------------

bool
MetalMusic3Ar::config_from_dir(const std::string& root, Config* out,
                               std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const std::filesystem::path r(root);
  FlexData lm, dd;
  if (!read_json_(r / "language_model" / "config.json", &lm)) {
    return fail("no language_model/config.json under " + root);
  }
  if (!read_json_(r / "rvq_depth_decoder" / "config.json", &dd)) {
    return fail("no rvq_depth_decoder/config.json under " + root);
  }
  auto lo = lm.as_object();
  auto dobj = dd.as_object();
  if (!lo.contains("model_type") || lo.at("model_type").as_string() !=
                                        "qwen3") {
    return fail("language_model/ is not a Qwen3 decoder");
  }
  if (!dobj.contains("_class_name") ||
      dobj.at("_class_name").as_string() != "MiniMaxMusic3RVQDepthDecoder") {
    return fail("rvq_depth_decoder/ is not MiniMaxMusic3RVQDepthDecoder");
  }
  auto gi = [](const auto& o, const char* k, int v) {
    return o.contains(k) ? (int)o.at(k).as_int(v) : v;
  };
  Config c;
  c.layers   = gi(lo, "num_hidden_layers", c.layers);
  c.hidden   = gi(lo, "hidden_size", c.hidden);
  c.heads    = gi(lo, "num_attention_heads", c.heads);
  c.kv_heads = gi(lo, "num_key_value_heads", c.kv_heads);
  c.head_dim = gi(lo, "head_dim", c.head_dim);
  c.ffn      = gi(lo, "intermediate_size", c.ffn);
  c.vocab    = gi(lo, "vocab_size", c.vocab);
  c.context  = gi(lo, "max_position_embeddings", c.context);
  if (lo.contains("rms_norm_eps")) {
    c.rms_eps = (float)lo.at("rms_norm_eps").as_real(c.rms_eps);
  }
  if (lo.contains("rope_parameters") && lo.at("rope_parameters").is_object()) {
    const FlexData rp = lo.at("rope_parameters");
    auto ro = rp.as_object();
    if (ro.contains("rope_theta")) {
      c.rope_theta = (float)ro.at("rope_theta").as_real(c.rope_theta);
    }
  } else if (lo.contains("rope_theta")) {
    c.rope_theta = (float)lo.at("rope_theta").as_real(c.rope_theta);
  }
  c.d_layers    = gi(dobj, "num_layers", c.d_layers);
  c.d_heads     = gi(dobj, "num_attention_heads", c.d_heads);
  c.d_ffn       = gi(dobj, "intermediate_size", c.d_ffn);
  c.d_vocab     = gi(dobj, "audio_vocab_size", c.d_vocab);
  c.codebooks   = gi(dobj, "num_codebooks", c.codebooks);
  c.d_positions = gi(dobj, "max_position_embeddings", c.d_positions);
  if (gi(dobj, "hidden_size", c.hidden) != c.hidden ||
      c.codebooks != music3::kCodebooks ||
      c.d_vocab != music3::kResidualCodes || c.d_positions > 16 ||
      c.hidden % c.d_heads != 0 || (c.hidden / c.d_heads) % 32 != 0 ||
      c.vocab < music3::kCodeOffset + music3::kSemanticCodes) {
    return fail("a MiniMax-Music3 LM / depth decoder shape this port does "
                "not run");
  }
  *out = c;
  return true;
}

std::size_t
MetalMusic3Ar::lm_weight_bytes(const std::string& root, int lm_quant_bits)
{
  const std::string dir =
      (std::filesystem::path(root) / "language_model").string();
  auto wts = MetalLlamaWeights::open_model(dir);
  Config c;
  if (!wts.has_value() || !config_from_dir(root, &c, nullptr)) { return 0; }
  std::size_t disk = 0;
  for (const std::string& nm : wts->tensor_names()) {
    if (const auto* ti = wts->info(nm)) { disk += (std::size_t)ti->nbytes; }
  }
  // The tables are not held whole: the head over the code window and the
  // code embeddings are.
  for (const char* t : {"lm_head.weight", "model.embed_tokens.weight"}) {
    if (const auto* ti = wts->info(t)) {
      disk -= std::min(disk, (std::size_t)ti->nbytes);
    }
  }
  disk += (std::size_t)(music3::kHeadRows + music3::kSemanticCodes) *
          (std::size_t)c.hidden * 2;
  if (lm_quant_bits != 8) { return disk; }
  // Built in memory, the layers are w8.
  const auto rs = MetalQwenModel::raw_sizes(*wts, "model.layers.", c.layers,
                                            8);
  if (rs.n_layers != c.layers || rs.pack || rs.raw > disk) { return disk; }
  return disk - rs.raw + rs.built;
}

std::size_t
MetalMusic3Ar::depth_weight_bytes(const std::string& root,
                                  int depth_quant_bits)
{
  const std::string dir =
      (std::filesystem::path(root) / "rvq_depth_decoder").string();
  auto wts = MetalLlamaWeights::open_model(dir);
  if (!wts.has_value()) { return 0; }
  // As lin_ builds them: the projections (each layer's q/k/v/out and
  // gate/up/down, and `projection`) w8 g64 with bf16 scales and biases,
  // every other tensor bf16 as stored.
  auto projection = [](const std::string& n) {
    if (n == "projection.weight") { return true; }
    if (n.rfind("layers.", 0) != 0) { return false; }
    for (const char* p : {".attn.to_q.", ".attn.to_k.", ".attn.to_v.",
                          ".attn.to_out.", ".gate_proj.", ".up_proj.",
                          ".down_proj."}) {
      if (n.find(p) != std::string::npos) { return true; }
    }
    return false;
  };
  std::size_t b = 0;
  for (const std::string& nm : wts->tensor_names()) {
    const auto* ti = wts->info(nm);
    if (ti == nullptr) { continue; }
    if (depth_quant_bits == 8 && projection(nm) && ti->shape.size() == 2) {
      const std::size_t N = (std::size_t)ti->shape[0];
      const std::size_t K = (std::size_t)ti->shape[1];
      b += N * K + 2 * N * (K / 64) * 2;
    } else {
      b += (std::size_t)ti->nbytes;
    }
  }
  return b;
}

// ---- load -----------------------------------------------------------------

MetalMusic3Ar::~MetalMusic3Ar() = default;

bool
MetalMusic3Ar::lm_quantized() const
{
  return _lm != nullptr && _lm->quantized_on_load();
}

std::unique_ptr<MetalMusic3Ar>
MetalMusic3Ar::load(std::shared_ptr<WeightSet> lm,
                    std::shared_ptr<WeightSet> depth, MetalCompute* mc,
                    const Config& cfg, std::string* err)
{
  if (lm == nullptr || depth == nullptr || mc == nullptr) {
    if (err != nullptr) { *err = "no checkpoint or no metal device"; }
    return nullptr;
  }
  std::unique_ptr<MetalMusic3Ar> m(new MetalMusic3Ar());
  if (!m->init_(lm, depth, mc, cfg, err)) { return nullptr; }
  return m;
}

MetalQwenModel::Config
MetalMusic3Ar::qwen_config_(const Config& cfg)
{
  MetalQwenModel::Config q;
  q.n_layers   = cfg.layers;
  q.hidden     = cfg.hidden;
  q.n_heads    = cfg.heads;
  q.n_kv_heads = cfg.kv_heads;
  q.head_dim   = cfg.head_dim;
  q.ffn_inner  = cfg.ffn;
  q.vocab      = cfg.vocab;
  q.rope_theta = cfg.rope_theta;
  q.rms_eps    = cfg.rms_eps;
  q.rotary_dim = cfg.head_dim;
  q.tie_embeddings     = false;
  q.use_bf16           = true;
  q.dense              = true;
  q.zero_centered_norm = false;
  q.quant_bits         = 8;
  q.load_quant_bits    = cfg.lm_quant_bits == 8 ? 8 : 0;
  q.attn_output_gate   = false;
  q.qk_norm            = true;
  q.backbone_only      = true;
  q.weight_prefix      = "";
  q.model_seg          = "model.";
  q.max_seq            = cfg.context + 256;
  // The pool's cap is the two contexts' own: each at most max_seq. Left
  // to its default (16 contexts' worth) the pool's doubling ran past what
  // two can fill -- 128 pages, 4.8 GB, for a six-minute song's 80.
  q.max_pages = 2 * ((q.max_seq + q.page_tokens - 1) / q.page_tokens);
  return q;
}

MetalMusic3Ar::Lin
MetalMusic3Ar::lin_(WeightSet& ws, const std::string& key,
                    const std::vector<std::string>& parts, int k,
                    bool interleave, bool quantize)
{
  Lin L;
  int n = 0;
  for (const auto& p : parts) {
    const auto* info = ws.src().info(p);
    if (info == nullptr || info->dtype != "BF16" || info->shape.size() != 2 ||
        (int)info->shape[1] != k) {
      return L;
    }
    n += (int)info->shape[0];
  }
  L.n = n;
  L.k = k;
  // The bf16 matrix, its parts' rows stacked -- or, `interleave`, rows of
  // the two parts alternating (gate f, up f) for swiglu_interleaved.
  auto build = [&]() -> SharedBuffer {
    SharedBuffer o = _mc->make_shared_buffer((std::size_t)n * k * 2);
    if (o.empty()) { return {}; }
    auto* d = static_cast<std::uint8_t*>(o.contents());
    const std::size_t row = (std::size_t)k * 2;
    std::size_t r0 = 0;
    for (std::size_t pi = 0; pi < parts.size(); ++pi) {
      SharedBuffer raw = ws.read(parts[pi], _mc, WeightSet::Residency::Copied);
      const std::size_t rows = raw.byte_size() / row;
      if (raw.empty()) { return {}; }
      const auto* s = static_cast<const std::uint8_t*>(raw.contents());
      for (std::size_t r = 0; r < rows; ++r) {
        const std::size_t dst =
            interleave ? (2 * r + pi) : (r0 + r);
        std::memcpy(d + dst * row, s + r * row, row);
      }
      r0 += rows;
    }
    return o;
  };
  if (!quantize) {
    L.w = ws.derived(std::string(kKey) + "bf16|" + key, build);
    return L;
  }
  // w8 g64, fit to bf16 scales (the model runs bf16). The three buffers
  // are built together; a later key reads the stash.
  SharedBuffer sq, ss, sb;
  int tried = 0;                    // 1 built, -1 failed
  auto quant = [&]() -> bool {
    if (tried != 0) { return tried > 0; }
    tried = -1;
    SharedBuffer dense = build();
    if (dense.empty()) { return false; }
    AffineQuantizer q(_mc);
    if (!q.valid() ||
        !q.quantize_linear(dense, /*src_bf16=*/true, n, k, 8, 64, 1.0f, sq,
                           ss, sb, AffineQuantizer::Scales::kFitBf16)) {
      return false;
    }
    tried = 1;
    return true;
  };
  const std::string base = std::string(kKey) + "w8|" + key;
  L.w = ws.derived(base + "|q", [&]() -> SharedBuffer {
    return quant() ? std::move(sq) : SharedBuffer{};
  });
  L.s = ws.derived(base + "|s", [&]() -> SharedBuffer {
    return quant() ? std::move(ss) : SharedBuffer{};
  });
  L.b = ws.derived(base + "|b", [&]() -> SharedBuffer {
    return quant() ? std::move(sb) : SharedBuffer{};
  });
  if (L.w.empty() || L.s.empty() || L.b.empty()) { return Lin{}; }
  return L;
}

bool
MetalMusic3Ar::init_(const std::shared_ptr<WeightSet>& lm,
                     const std::shared_ptr<WeightSet>& depth,
                     MetalCompute* mc, const Config& cfg, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  _lm_ws = lm;
  _depth_ws = depth;
  _mc = mc;
  _cfg = cfg;
  const std::size_t H = (std::size_t)cfg.hidden;

  // ---- the global LM: a plain Qwen3 dense stack --------------------------
  const MetalQwenModel::Config q = qwen_config_(cfg);
  _lm = MetalQwenModel::load(lm, mc, q);
  if (!_lm) { return fail("the MiniMax-Music3 LM did not load"); }

  // The head over the code window and the code embeddings, read as
  // ranges of the two 200000-row tables.
  {
    std::string rerr;
    _rows = SafetensorsRanges::open(
        lm->dir(), lm->src(), &rerr, /*cached=*/true,
        [](std::string_view n) {
          return n == "lm_head.weight" || n == "model.embed_tokens.weight";
        });
    if (!_rows) { return fail("cannot read the LM's tables: " + rerr); }
  }
  auto window = [&](const char* tensor, std::int32_t row0, int rows,
                    const char* key) -> SharedBuffer {
    return lm->derived(std::string(kKey) + key, [&]() -> SharedBuffer {
      SharedBuffer o = mc->make_shared_buffer((std::size_t)rows * H * 2);
      if (o.empty()) { return {}; }
      if (!_rows->read(tensor, (std::size_t)row0 * H * 2,
                       (std::size_t)rows * H * 2, o.contents())) {
        return {};
      }
      return o;
    });
  };
  _head = window("lm_head.weight", music3::kHeadRow0, music3::kHeadRows,
                 "head-window");
  _code_emb = window("model.embed_tokens.weight", music3::kCodeOffset,
                     music3::kSemanticCodes, "code-embeddings");
  if (_head.empty() || _code_emb.empty()) {
    return fail("the LM's head / code embedding windows did not read");
  }

  // ---- the depth decoder ---------------------------------------------------
  WeightSet& d = *depth;
  const bool w8 = cfg.depth_quant_bits == 8;
  _depth_w8 = w8;
  auto t = [&](const std::string& nm) {
    return d.tensor(nm, mc, WeightSet::Residency::Copied);
  };
  const int k = cfg.hidden;
  _d.resize((std::size_t)cfg.d_layers);
  for (int L = 0; L < cfg.d_layers; ++L) {
    const std::string p = "layers." + std::to_string(L) + ".";
    DLayer& y = _d[(std::size_t)L];
    y.ln1 = t(p + "input_layernorm.weight");
    y.ln2 = t(p + "post_attention_layernorm.weight");
    y.qkv = lin_(d, p + "qkv",
                 {p + "attn.to_q.weight", p + "attn.to_k.weight",
                  p + "attn.to_v.weight"}, k, false, w8);
    y.out = lin_(d, p + "out", {p + "attn.to_out.weight"}, k, false, w8);
    y.gu = lin_(d, p + "gu", {p + "gate_proj.weight", p + "up_proj.weight"},
                k, true, w8);
    y.down = lin_(d, p + "down", {p + "down_proj.weight"}, cfg.d_ffn, false,
                  w8);
    if (y.ln1.empty() || y.ln2.empty() || y.qkv.empty() || y.out.empty() ||
        y.gu.empty() || y.down.empty() || y.qkv.n != 3 * k ||
        y.gu.n != 2 * cfg.d_ffn || y.down.n != k) {
      return fail("depth decoder layer " + std::to_string(L) +
                  " is incomplete");
    }
  }
  _proj = lin_(d, "proj", {"projection.weight"}, k, false, w8);
  _pos = t("pos_embedding.weight");
  _dnorm = t("norm.weight");
  _aemb = t("audio_embeddings.weight");
  // The seven heads, one [7 * 1024, H] bf16 matrix.
  {
    std::vector<std::string> heads;
    for (int i = 0; i + 1 < cfg.codebooks; ++i) {
      heads.push_back("audio_heads." + std::to_string(i) + ".weight");
    }
    _dheads = lin_(d, "heads", heads, k, false, false).w;
  }
  if (_proj.empty() || _pos.empty() || _dnorm.empty() || _aemb.empty() ||
      _dheads.empty()) {
    return fail("the depth decoder is incomplete");
  }

  // ---- kernels ------------------------------------------------------------
  _lib_gemm = mc->load_library("dense_gemm_bf16");
  _lib_qmv  = mc->load_library("affine_qmv_bf16");
  _lib_rms  = mc->load_library("rms_norm_bf16");
  _lib_elt  = mc->load_library("llm_elementwise_bf16");
  _lib_m3   = mc->load_library("minimax_music3_bf16");
  _fn_gemv_rows = _lib_gemm.function("dense_gemv_rows_t_f16");
  _fn_qmv_batch = _lib_qmv.function("affine_qmv_batch_w8g64");
  _fn_qmv_batch4 = _lib_qmv.function("affine_qmv_batch4_w8g64");
  _fn_rms  = _lib_rms.function("rms_norm_f16");
  _fn_add  = _lib_elt.function("residual_add_f16");
  _fn_glu  = _lib_elt.function("swiglu_interleaved_f16");
  _fn_copy = _lib_elt.function("copy_f16");
  _fn_kv   = _lib_m3.function("music3_depth_kv_append");
  _fn_attn = _lib_m3.function("music3_depth_attn");
  _fn_pos  = _lib_m3.function("music3_depth_add_pos");
  if (!_fn_gemv_rows.valid() || !_fn_qmv_batch.valid() ||
      !_fn_qmv_batch4.valid() || !_fn_rms.valid() ||
      !_fn_add.valid() || !_fn_glu.valid() || !_fn_copy.valid() ||
      !_fn_kv.valid() || !_fn_attn.valid() || !_fn_pos.valid()) {
    return fail("a MiniMax-Music3 AR kernel did not resolve");
  }

  // ---- scratch (small: a step is at most 4 rows; own_scratch_bytes_) -----
  const std::size_t R = 4;
  auto mk = [&](std::size_t elts) { return mc->make_shared_buffer(elts * 2); };
  _h = mk(2 * H);
  _logits = mk(2 * (std::size_t)music3::kHeadRows);
  _emb = mk(2 * H);
  _din = mk(R * H);
  _dx = mk(R * H);
  _dn = mk(R * H);
  _dqkv = mk(R * 3 * H);
  _datt = mk(R * H);
  _dtmp = mk(R * H);
  _dgu = mk(R * 2 * (std::size_t)cfg.d_ffn);
  _dact = mk(R * (std::size_t)cfg.d_ffn);
  _dlogits = mk(R * (std::size_t)cfg.d_vocab);
  // Every layer's K/V for both guidance rows.
  const std::size_t kvc = (std::size_t)cfg.d_layers * 2 * cfg.d_positions * H;
  _kc = mk(kvc);
  _vc = mk(kvc);
  for (const SharedBuffer* s : {&_h, &_logits, &_emb, &_din, &_dx, &_dn,
                                &_dqkv, &_datt, &_dtmp, &_dgu, &_dact,
                                &_dlogits, &_kc, &_vc}) {
    if (s->empty()) { return fail("AR scratch allocation failed"); }
  }
  return true;
}

std::size_t
MetalMusic3Ar::resident_bytes() const
{
  std::size_t b = _lm ? (std::size_t)_lm->resident_bytes() : 0;
  b += _head.byte_size() + _code_emb.byte_size();
  auto lin = [](const Lin& l) {
    return l.w.byte_size() + l.s.byte_size() + l.b.byte_size();
  };
  for (const DLayer& y : _d) {
    b += y.ln1.byte_size() + y.ln2.byte_size() + lin(y.qkv) + lin(y.out) +
         lin(y.gu) + lin(y.down);
  }
  b += lin(_proj) + _pos.byte_size() + _dnorm.byte_size() +
       _aemb.byte_size() + _dheads.byte_size();
  return b;
}

std::size_t
MetalMusic3Ar::depth_bytes() const
{
  auto lin = [](const Lin& l) {
    return l.w.byte_size() + l.s.byte_size() + l.b.byte_size();
  };
  std::size_t b = 0;
  for (const DLayer& y : _d) {
    b += y.ln1.byte_size() + y.ln2.byte_size() + lin(y.qkv) + lin(y.out) +
         lin(y.gu) + lin(y.down);
  }
  return b + lin(_proj) + _pos.byte_size() + _dnorm.byte_size() +
         _aemb.byte_size() + _dheads.byte_size();
}

std::size_t
MetalMusic3Ar::held_scratch_bytes() const
{
  if (!_lm) { return 0; }
  ContextManager* cm = _lm->context_manager();
  return cm != nullptr ? (std::size_t)cm->resident_bytes() : 0;
}

std::size_t
MetalMusic3Ar::own_scratch_bytes_(const Config& c)
{
  const std::size_t H = (std::size_t)c.hidden, R = 4;
  const std::size_t F = (std::size_t)c.d_ffn;
  std::size_t e = 2 * H + 2 * (std::size_t)music3::kHeadRows + 2 * H;
  e += 5 * R * H + R * 3 * H + R * 2 * F + R * F +
       R * (std::size_t)c.d_vocab;
  e += 2 * (std::size_t)c.d_layers * 2 * c.d_positions * H;   // K/V
  return e * 2;
}

std::size_t
MetalMusic3Ar::run_bytes(const Config& c, int prefix, int frames,
                         bool matrix_cores)
{
  // Each context holds its prompt and every frame, the first included,
  // within the model's context.
  const std::int64_t per = std::min<std::int64_t>(
      c.context, (std::int64_t)std::max(prefix, 1) + std::max(frames, 0) + 1);
  const MetalQwenModel::Config q = qwen_config_(c);
  // Both contexts' K/V; each prompt prefilled on its own.
  std::size_t b = MetalQwenModel::run_bytes(
      q, 2 * per, 2, /*one_pass=*/false,
      /*plain_mlp=*/c.lm_quant_bits != 8, matrix_cores,
      /*prompt_rows=*/std::min<std::int64_t>(std::max(prefix, 1),
                                             music3::kMaxPromptTokens));
  // The batched step's scratch at two rows (ensure_bscratch_: the logits
  // row and the page table are the bulk of it).
  b += 2 * ((std::size_t)c.vocab + 8 * (std::size_t)c.ffn +
            16 * (std::size_t)c.hidden) * 2 +
       2 * (std::size_t)q.max_seq * 3 * 4;
  return b + own_scratch_bytes_(c);
}

std::size_t
MetalMusic3Ar::handoff_bytes(const Config& c, int frames)
{
  return (std::size_t)std::max(frames, 0) * (std::size_t)c.codebooks *
         (std::size_t)c.hidden * 2;
}

// ---- pieces -----------------------------------------------------------------

SharedBuffer
MetalMusic3Ar::embed_prompt_(const std::vector<std::int32_t>& ids,
                             std::string* err)
{
  const std::size_t H = (std::size_t)_cfg.hidden;
  SharedBuffer x = _mc->make_shared_buffer(ids.size() * H * 2);
  if (x.empty()) { return {}; }
  auto* d = static_cast<std::uint8_t*>(x.contents());
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const std::int32_t id = ids[i];
    if (id < 0 || id >= _cfg.vocab ||
        !_rows->read("model.embed_tokens.weight", (std::size_t)id * H * 2,
                     H * 2, d + i * H * 2)) {
      if (err != nullptr) { *err = "a prompt row did not read"; }
      return {};
    }
  }
  return x;
}

void
MetalMusic3Ar::gemv_(ComputeEncoder& enc, const Lin& w, const SharedBuffer& x,
                     std::size_t xe, const SharedBuffer& y, std::size_t ye,
                     int rows)
{
  if (w.w8()) {
    // Weights read once: two rows a tile, or a frame's first step's four
    // in one (the same per-row arithmetic either way).
    const bool four = rows > 2;
    enc.set_function(four ? _fn_qmv_batch4 : _fn_qmv_batch);
    enc.set_buffer(0, w.w);
    enc.set_buffer(1, w.s);
    enc.set_buffer(2, w.b);
    enc.set_buffer(3, x, xe * 2);
    enc.set_buffer(4, y, ye * 2);
    enc.set_constant(5, w.k);
    enc.set_constant(6, w.n);
    enc.set_constant(7, rows);
    enc.dispatch({32, (unsigned)(w.n / 4),
                  (unsigned)(four ? (rows + 3) / 4 : (rows + 1) / 2)},
                 {32, 2, 1});
    return;
  }
  enc.set_function(_fn_gemv_rows);
  enc.set_buffer(0, x, xe * 2);
  enc.set_buffer(1, w.w);
  enc.set_buffer(2, y, ye * 2);
  enc.set_constant(3, w.k);
  enc.set_constant(4, w.n);
  enc.dispatch({32u * (unsigned)rows, (unsigned)(((w.n + 7) / 8) * 2), 1},
               {32, 2, 1});
}

void
MetalMusic3Ar::encode_head_(ComputeEncoder& enc, int row)
{
  const int H = _cfg.hidden;
  enc.set_function(_fn_gemv_rows);
  enc.set_buffer(0, _h, (std::size_t)row * H * 2);
  enc.set_buffer(1, _head);
  enc.set_buffer(2, _logits, (std::size_t)row * music3::kHeadRows * 2);
  enc.set_constant(3, H);
  enc.set_constant(4, music3::kHeadRows);
  enc.dispatch({32, (unsigned)(((music3::kHeadRows + 7) / 8) * 2), 1},
               {32, 2, 1});
}

CommandStream&
MetalMusic3Ar::stream_()
{
  static const bool fresh =
      std::getenv("VPIPE_MUSIC3_FRESH_STREAMS") != nullptr;
  if (fresh || !_stream.valid()) { _stream = _mc->make_command_stream(); }
  return _stream;
}

bool
MetalMusic3Ar::depth_step_(int i, int prev_code, _Float16* frame_hidden,
                           std::string* err)
{
  const int H = _cfg.hidden;
  const int Dh = H / _cfg.d_heads;
  const int F = _cfg.d_ffn;
  const int P = _cfg.d_positions;
  const float eps = _cfg.rms_eps;
  // Step 1: positions 0 and 1 of both rows (4 rows: b0p0, b0p1, b1p0,
  // b1p1); later steps position i of both rows.
  const int np = i == 1 ? 2 : 1;
  const int p0 = i == 1 ? 0 : i;
  const int R = 2 * np;
  {
    auto* in = static_cast<std::uint16_t*>(_din.contents());
    const auto* h = static_cast<const std::uint16_t*>(_h.contents());
    for (int b = 0; b < 2; ++b) {
      if (i == 1) {
        std::memcpy(in + (std::size_t)(2 * b) * H, h + (std::size_t)b * H,
                    (std::size_t)H * 2);
        std::memcpy(in + (std::size_t)(2 * b + 1) * H,
                    static_cast<const std::uint16_t*>(_code_emb.contents()) +
                        (std::size_t)prev_code * H,
                    (std::size_t)H * 2);
      } else {
        // c_{i-1}'s own embedding, its codebook's band of the table.
        const std::size_t row =
            (std::size_t)(prev_code + (i - 2) * _cfg.d_vocab);
        std::memcpy(in + (std::size_t)b * H,
                    static_cast<const std::uint16_t*>(_aemb.contents()) +
                        row * H,
                    (std::size_t)H * 2);
      }
    }
  }
  auto rms = [&](ComputeEncoder& enc, const SharedBuffer& x,
                 const SharedBuffer& w, const SharedBuffer& y) {
    enc.set_function(_fn_rms);
    enc.set_buffer(0, x);
    enc.set_buffer(1, w);
    enc.set_buffer(2, y);
    enc.set_constant(3, H);
    enc.set_constant(4, eps);
    enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
  };
  auto add = [&](ComputeEncoder& enc, const SharedBuffer& a,
                 const SharedBuffer& b) {
    const int n = R * H;
    enc.set_function(_fn_add);
    enc.set_buffer(0, a);
    enc.set_buffer(1, b);
    enc.set_buffer(2, a);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };
  CommandStream& s = stream_();
  {
    ComputeEncoder enc = s.begin_compute();
    gemv_(enc, _proj, _din, 0, _dx, 0, R);
    enc.set_function(_fn_pos);
    enc.set_buffer(0, _dx);
    enc.set_buffer(1, _pos);
    enc.set_constant(2, H);
    enc.set_constant(3, R);
    enc.set_constant(4, np);
    enc.set_constant(5, p0);
    enc.dispatch({(unsigned)H, (unsigned)R, 1}, {256, 1, 1});
    const float scale = 1.0f / std::sqrt((float)Dh);
    for (std::size_t L = 0; L < _d.size(); ++L) {
      const DLayer& y = _d[L];
      const std::size_t cache = L * 2 * (std::size_t)P * H;
      rms(enc, _dx, y.ln1, _dn);
      gemv_(enc, y.qkv, _dn, 0, _dqkv, 0, R);
      enc.set_function(_fn_kv);
      enc.set_buffer(0, _dqkv);
      enc.set_buffer(1, _kc, cache * 2);
      enc.set_buffer(2, _vc, cache * 2);
      enc.set_constant(3, H);
      enc.set_constant(4, R);
      enc.set_constant(5, np);
      enc.set_constant(6, p0);
      enc.set_constant(7, P);
      enc.dispatch({(unsigned)H, (unsigned)R, 1}, {256, 1, 1});
      enc.set_function(_fn_attn);
      enc.set_buffer(0, _dqkv);
      enc.set_buffer(1, _kc, cache * 2);
      enc.set_buffer(2, _vc, cache * 2);
      enc.set_buffer(3, _datt);
      enc.set_constant(4, H);
      enc.set_constant(5, Dh);
      enc.set_constant(6, np);
      enc.set_constant(7, p0);
      enc.set_constant(8, P);
      enc.set_constant(9, scale);
      enc.dispatch({(unsigned)(Dh * R), (unsigned)_cfg.d_heads, 1},
                   {(unsigned)Dh, 1, 1});
      gemv_(enc, y.out, _datt, 0, _dtmp, 0, R);
      add(enc, _dx, _dtmp);
      rms(enc, _dx, y.ln2, _dn);
      gemv_(enc, y.gu, _dn, 0, _dgu, 0, R);
      enc.set_function(_fn_glu);
      enc.set_buffer(0, _dgu);
      enc.set_buffer(1, _dact);
      enc.set_constant(2, R);
      enc.set_constant(3, F);
      enc.dispatch({(unsigned)(R * F), 1, 1}, {256, 1, 1});
      gemv_(enc, y.down, _dact, 0, _dtmp, 0, R);
      add(enc, _dx, _dtmp);
    }
    rms(enc, _dx, _dnorm, _dn);
    // Head i-1 over every row (the earlier positions' rows are discarded).
    enc.set_function(_fn_gemv_rows);
    enc.set_buffer(0, _dn);
    enc.set_buffer(1, _dheads, (std::size_t)(i - 1) * _cfg.d_vocab * H * 2);
    enc.set_buffer(2, _dlogits);
    enc.set_constant(3, H);
    enc.set_constant(4, _cfg.d_vocab);
    enc.dispatch({32u * (unsigned)R, (unsigned)(((_cfg.d_vocab + 7) / 8) * 2),
                  1}, {32, 2, 1});
  }
  std::string gerr;
  if (!s.commit().wait_ok(&gerr)) {
    if (err != nullptr) {
      *err = gerr.empty() ? "depth decoder step failed" : gerr;
    }
    return false;
  }
  // The conditional row's normed hidden at its last position: hiddens
  // slot i of the frame (bf16 -> f16).
  if (frame_hidden != nullptr) {
    const auto* n = static_cast<const std::uint16_t*>(_dn.contents());
    const std::uint16_t* row = n + (std::size_t)(np - 1) * H;
    _Float16* dst = frame_hidden + (std::size_t)i * H;
    for (int c = 0; c < H; ++c) { dst[c] = (_Float16)bf16_f32_(row[c]); }
  }
  return true;
}

// ---- the loop ---------------------------------------------------------------

bool
MetalMusic3Ar::generate(const std::vector<std::int32_t>& ids, int max_frames,
                        std::uint64_t seed, const FrameFn& on_frame,
                        Song* out, std::string* err, float cfg)
{
  return run_(ids, max_frames, seed, cfg, on_frame, nullptr, out, nullptr,
              err);
}

bool
MetalMusic3Ar::generate_forced(const std::vector<std::int32_t>& ids,
                               int max_frames,
                               const std::vector<std::array<int, 8>>& forced,
                               Song* out, Forced* rec, std::string* err)
{
  return run_(ids, max_frames, 0, music3::kArCfg, {}, &forced, out, rec,
              err);
}

bool
MetalMusic3Ar::run_(const std::vector<std::int32_t>& ids, int max_frames,
                    std::uint64_t seed, float cfg, const FrameFn& on_frame,
                    const std::vector<std::array<int, 8>>* forced,
                    Song* out, Forced* rec, std::string* err)
{
  using clk = std::chrono::steady_clock;
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const auto t0 = clk::now();
  *out = Song{};
  if (ids.size() < 3) { return fail("an empty prompt"); }
  if ((int)ids.size() > music3::kMaxPromptTokens) {
    return fail(fmt("the prompt is {} tokens; the most is {}", ids.size(),
                    music3::kMaxPromptTokens)());
  }
  // Room in the context: the prompt, the first (advancing) frame and the
  // kept ones.
  max_frames = std::min({max_frames, music3::kMaxFrames,
                         _cfg.context - (int)ids.size() - 1});
  if (max_frames < 1) { return fail("no room in the context for a frame"); }
  const int H = _cfg.hidden;
  const std::size_t row = (std::size_t)_cfg.codebooks * H;
  const SessionContextIntf* session = _mc->session();

  ContextManager* cm = _lm->context_manager();
  const ContextId cc = cm->acquire_root();
  const ContextId uc = cm->acquire_root();
  struct Release {
    ContextManager* cm; ContextId a, b;
    ~Release() { cm->release(a); cm->release(b); }
  } rel{cm, cc, uc};

  // ---- prefill both prompts --------------------------------------------
  {
    PerfAuxScope _perf(session, kPerfLaneLLM, kGvidLlmPrefill,
                       kPerfLlmPrefillBegin, (std::uint64_t)ids.size());
    const std::vector<std::int32_t> uids = music3::unconditional_ids(ids);
    for (int b = 0; b < 2; ++b) {
      SharedBuffer x = embed_prompt_(b == 0 ? ids : uids, err);
      if (x.empty()) { return false; }
      SharedBuffer h = _lm->forward_embeddings_hidden(b == 0 ? cc : uc, x,
                                                      (int)ids.size());
      if (h.empty()) { return fail("the LM's prefill failed"); }
      std::memcpy(static_cast<std::uint8_t*>(_h.contents()) +
                      (std::size_t)b * H * 2,
                  h.contents(), (std::size_t)H * 2);
    }
    CommandStream& s = stream_();
    {
      ComputeEncoder enc = s.begin_compute();
      encode_head_(enc, 0);
      encode_head_(enc, 1);
    }
    if (!s.commit().wait_ok()) { return fail("the LM's head failed"); }
  }
  out->prefill_seconds =
      std::chrono::duration<double>(clk::now() - t0).count();

  music3::FramePicker picker(seed, cfg);
  // VPIPE_MUSIC3_SERIAL_LM=1 steps the two contexts one after the other
  // (the A/B of the batched step).
  // A bf16 stack has no batched forward and always steps them so.
  static const bool serial_env =
      std::getenv("VPIPE_MUSIC3_SERIAL_LM") != nullptr;
  // VPIPE_MUSIC3_AR_PROFILE=1: a frame's wall time split into the LM
  // step, the depth steps and the rest, printed at the end.
  static const bool prof = std::getenv("VPIPE_MUSIC3_AR_PROFILE") != nullptr;
  double t_lm = 0.0, t_depth = 0.0;
  int n_prof = 0;
  const bool serial = serial_env || !_lm->quantized_on_load();
  std::vector<float> lc((std::size_t)music3::kHeadRows),
      lu((std::size_t)music3::kHeadRows);
  std::vector<float> dc((std::size_t)_cfg.d_vocab), du(dc.size());
  std::vector<_Float16> fh(row);
  std::vector<float> fb((std::size_t)H);
  const float frame_scale = 1.0f / std::sqrt((float)_cfg.codebooks);
  const std::uint16_t* code_emb =
      static_cast<const std::uint16_t*>(_code_emb.contents());
  const std::uint16_t* aemb =
      static_cast<const std::uint16_t*>(_aemb.contents());
  auto widen = [](const SharedBuffer& b, std::size_t off, std::size_t n,
                  float* dst) {
    const auto* s = static_cast<const std::uint16_t*>(b.contents()) + off;
    for (std::size_t i = 0; i < n; ++i) { dst[i] = bf16_f32_(s[i]); }
  };

  // The first decode step only advances past <|audio_start|>: its codes
  // are fed back, its hidden states not kept.
  for (int fi = 0; fi <= max_frames; ++fi) {
    if (forced != nullptr && fi >= (int)forced->size()) { break; }
    PerfAuxScope _perf(session, kPerfLaneLLM, kGvidLlmDecode,
                       kPerfLlmDecodeBegin, 1);
    widen(_logits, 0, lc.size(), lc.data());
    widen(_logits, lc.size(), lu.size(), lu.data());
    if (rec != nullptr) {
      rec->c0_logits.insert(rec->c0_logits.end(), lc.begin(), lc.end());
      rec->c0_logits.insert(rec->c0_logits.end(), lu.begin(), lu.end());
    }
    int c0 = -1;
    if (forced != nullptr) {
      c0 = (*forced)[(std::size_t)fi][0];
    } else {
      const int w = picker.pick_semantic(lc.data(), lu.data());
      c0 = w == 0 ? -1 : w - music3::kSemanticBase;
    }
    if (c0 < 0) {
      out->ended = true;
      if (rec != nullptr) {
        rec->depth_logits.resize(rec->depth_logits.size() +
                                 (std::size_t)7 * 2 * _cfg.d_vocab, 0.0f);
      }
      break;
    }
    // The frame hidden's slot 0: the conditional LM hidden that picked c0.
    {
      const auto* h = static_cast<const std::uint16_t*>(_h.contents());
      for (int c = 0; c < H; ++c) {
        fh[(std::size_t)c] = (_Float16)bf16_f32_(h[c]);
      }
    }
    std::array<int, 8> codes{};
    codes[0] = c0;
    int prev = c0;
    const auto td0 = clk::now();
    for (int i = 1; i < _cfg.codebooks; ++i) {
      if (!depth_step_(i, prev, fh.data(), err)) { return false; }
      // The logits of the LAST position of each row.
      const int np = i == 1 ? 2 : 1;
      widen(_dlogits, (std::size_t)(np - 1) * _cfg.d_vocab, dc.size(),
            dc.data());
      widen(_dlogits, (std::size_t)(2 * np - 1) * _cfg.d_vocab, du.size(),
            du.data());
      if (rec != nullptr) {
        rec->depth_logits.insert(rec->depth_logits.end(), dc.begin(),
                                 dc.end());
        rec->depth_logits.insert(rec->depth_logits.end(), du.begin(),
                                 du.end());
      }
      const int code = forced != nullptr
                           ? (*forced)[(std::size_t)fi][(std::size_t)i]
                           : picker.pick_residual(dc.data(), du.data());
      codes[(std::size_t)i] = code;
      prev = code;
    }
    if (prof) {
      t_depth += std::chrono::duration<double, std::milli>(clk::now() - td0)
                     .count();
      ++n_prof;
    }
    out->codes.push_back(codes);
    if (fi > 0) {
      out->hiddens.insert(out->hiddens.end(), fh.begin(), fh.end());
      ++out->frames;
      if (on_frame && !on_frame(out->frames)) {
        out->stopped = true;
        break;
      }
      if (out->frames >= max_frames) { break; }
    }
    // The feedback: the sum of the frame's eight embeddings / sqrt(8).
    {
      const std::uint16_t* e0 = code_emb + (std::size_t)c0 * H;
      for (int c = 0; c < H; ++c) { fb[(std::size_t)c] = bf16_f32_(e0[c]); }
      std::vector<float> extra((std::size_t)H, 0.0f);
      for (int i = 1; i < _cfg.codebooks; ++i) {
        const std::uint16_t* a =
            aemb + (std::size_t)(codes[(std::size_t)i] +
                                 (i - 1) * _cfg.d_vocab) * H;
        for (int c = 0; c < H; ++c) {
          extra[(std::size_t)c] += bf16_f32_(a[c]);
        }
      }
      auto* e = static_cast<std::uint16_t*>(_emb.contents());
      for (int c = 0; c < H; ++c) {
        e[c] = f32_bf16_((fb[(std::size_t)c] + extra[(std::size_t)c]) *
                         frame_scale);
      }
      std::memcpy(e + H, e, (std::size_t)H * 2);
    }
    const auto tl0 = clk::now();
    struct LmTime {
      bool on; double* acc; clk::time_point t0;
      ~LmTime() {
        if (on) {
          *acc += std::chrono::duration<double, std::milli>(clk::now() - t0)
                      .count();
        }
      }
    } lm_time{prof, &t_lm, tl0};
    if (!serial) {
      // Both contexts in one forward: every LM weight read once a frame.
      const std::function<void(ComputeEncoder&, const SharedBuffer&)> post =
          [&](ComputeEncoder& enc, const SharedBuffer& hn) {
            enc.set_function(_fn_copy);
            enc.set_buffer(0, hn);
            enc.set_buffer(1, _h);
            enc.set_constant(2, 0);
            enc.set_constant(3, 2 * H);
            enc.dispatch({(unsigned)(2 * H), 1, 1}, {256, 1, 1});
            encode_head_(enc, 0);
            encode_head_(enc, 1);
          };
      const ContextId both[2] = {cc, uc};
      if (_lm->decode_batched_embedding_hidden(both, _emb, &post) ==
          nullptr) {
        return fail("the LM's decode step failed");
      }
      continue;
    }
    for (int b = 0; b < 2; ++b) {
      const std::function<void(ComputeEncoder&, const SharedBuffer&)> post =
          [&](ComputeEncoder& enc, const SharedBuffer& hn) {
            enc.set_function(_fn_copy);
            enc.set_buffer(0, hn);
            enc.set_buffer(1, _h);
            enc.set_constant(2, b * H);
            enc.set_constant(3, H);
            enc.dispatch({(unsigned)H, 1, 1}, {256, 1, 1});
            encode_head_(enc, b);
          };
      const SharedBuffer* hn = _lm->decode_embedding_hidden(
          b == 0 ? cc : uc, _emb, -1, &post);
      if (hn == nullptr) { return fail("the LM's decode step failed"); }
    }
  }
  out->seconds = std::chrono::duration<double>(clk::now() - t0).count();
  if (prof && n_prof > 0) {
    const double per = 1e3 * (out->seconds - out->prefill_seconds) / n_prof;
    std::printf("[music3-ar-prof] %d frames: %.1f ms a frame = LM %.1f + "
                "depth %.1f (%.2f a step) + host %.1f\n", n_prof, per,
                t_lm / n_prof, t_depth / n_prof,
                t_depth / n_prof / (_cfg.codebooks - 1),
                per - (t_lm + t_depth) / n_prof);
  }
  return true;
}

}  // namespace vpipe::genai
