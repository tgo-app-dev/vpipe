#include "generative-models/drama3/metal-drama3-model.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "generative-models/context-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>

namespace vpipe::genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

float bf16_f32(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// The main sampler's candidate cap and repetition window (the reference's
// max_top_k and RAS_WINDOW_FRAMES).
constexpr int kMaxRank = 64;
constexpr int kRasWindow = 16;

}  // namespace

// ---- config ---------------------------------------------------------------

bool
Drama3Config::from_dir(const std::string& dir, Drama3Config* out,
                       std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  std::ifstream in(std::filesystem::path(dir) / "config.json");
  if (!in) { return fail("no config.json under " + dir); }
  FlexData fd;
  try {
    fd = FlexData::from_json(in);
  } catch (const std::exception& e) {
    return fail(std::string("config.json: ") + e.what());
  }
  if (!fd.is_object()) { return fail("config.json is not an object"); }
  const FlexData& root = fd;
  auto obj = root.as_object();
  auto str = [](const auto& o, const char* k) -> std::string {
    return o.contains(k) && o.at(k).is_string()
               ? std::string(o.at(k).as_string()) : std::string();
  };
  if (str(obj, "model_type") != "fish_omni") {
    return fail("not a Fish Audio Drama 3 checkpoint (model_type != "
                "\"fish_omni\")");
  }
  if (!obj.contains("text_config") || !obj.contains("audio_decoder_config")) {
    return fail("config.json has no text_config / audio_decoder_config");
  }
  const std::string tmpl = str(obj, "speech_template_version");
  if (!tmpl.empty() && tmpl != "reference-bracket-v1") {
    return fail("speech_template_version \"" + tmpl + "\" is not "
                "reference-bracket-v1, the one prompt layout this port "
                "builds");
  }
  const FlexData tcfd = obj.at("text_config");
  const FlexData dcfd = obj.at("audio_decoder_config");
  auto tc = tcfd.as_object();
  auto dc = dcfd.as_object();
  auto geti = [](const auto& o, const char* k, int d) {
    return o.contains(k) && !o.at(k).is_null() ? (int)o.at(k).as_int(d) : d;
  };
  auto getf = [](const auto& o, const char* k, float d) {
    return o.contains(k) && !o.at(k).is_null() ? (float)o.at(k).as_real(d)
                                               : d;
  };
  auto getb = [](const auto& o, const char* k, bool d) {
    return o.contains(k) && o.at(k).is_bool() ? o.at(k).as_bool(d) : d;
  };
  Drama3Config c;
  c.eos = geti(obj, "eos_token_id", c.eos);
  c.semantic_begin = geti(obj, "semantic_start_token_id", c.semantic_begin);
  c.semantic_end = geti(obj, "semantic_end_token_id", c.semantic_end);

  // ---- the backbone: Qwen3.5-MoE, every layer MoE ------------------------
  MetalQwenModel::Config& q = c.backbone;
  q.n_layers   = geti(tc, "num_hidden_layers", 40);
  q.hidden     = geti(tc, "hidden_size", 2048);
  q.n_heads    = geti(tc, "num_attention_heads", 16);
  q.n_kv_heads = geti(tc, "num_key_value_heads", 2);
  q.head_dim   = geti(tc, "head_dim", 256);
  q.ffn_inner  = 0;
  q.vocab      = geti(tc, "vocab_size", 252288);
  q.rms_eps    = getf(tc, "rms_norm_eps", 1e-6f);
  c.text_vocab = q.vocab;
  float theta = 1.0e7f, partial = 0.25f;
  if (tc.contains("rope_parameters") &&
      tc.at("rope_parameters").is_object()) {
    const FlexData rpfd = tc.at("rope_parameters");
    auto rp = rpfd.as_object();
    theta = getf(rp, "rope_theta", theta);
    partial = getf(rp, "partial_rotary_factor", partial);
    const std::string kind = str(rp, "rope_type");
    if (!kind.empty() && kind != "default") {
      return fail("text_config rope_type \"" + kind + "\" is not the "
                  "default rotary this port runs");
    }
  }
  q.rope_theta = theta;
  q.rotary_dim = (int)((float)q.head_dim * partial);
  if (getb(tc, "rotary_interleaved", false)) {
    return fail("an interleaved-rotary backbone is not the Qwen3.5 layout");
  }
  if (!getb(tc, "attn_output_gate", true) || !getb(tc, "use_qk_norm", true) ||
      !getb(tc, "layernorm_zero_centered_gamma", true) ||
      !getb(tc, "shared_expert_gate", true)) {
    return fail("text_config departs from Qwen3.5-MoE (output gate, q/k "
                "norm, zero-centred norms, gated shared expert)");
  }
  // Every layer MoE, full attention every 4th (layer_types names them).
  int interval = 4;
  if (tc.contains("layer_types") && tc.at("layer_types").is_array()) {
    const FlexData lt = tc.at("layer_types");
    auto a = lt.as_array();
    int first_full = -1;
    for (std::size_t i = 0; i < a.size(); ++i) {
      if (a[i].as_string() == "full_attention") {
        first_full = (int)i;
        break;
      }
    }
    if (first_full >= 0) { interval = first_full + 1; }
    for (std::size_t i = 0; i < a.size(); ++i) {
      const bool full = a[i].as_string() == "full_attention";
      if (full != (((int)i % interval) == interval - 1)) {
        return fail("layer_types is not a regular full-attention interval");
      }
    }
  }
  q.full_attn_interval = interval;
  if (tc.contains("mlp_layer_types") && tc.at("mlp_layer_types").is_array()) {
    const FlexData mt = tc.at("mlp_layer_types");
    for (const auto& v : mt.as_array()) {
      if (v.as_string() != "moe") {
        return fail("every backbone layer must be MoE");
      }
    }
  }
  q.gdn_conv_kernel = geti(tc, "linear_conv_kernel_dim", 4);
  q.gdn_k_heads = geti(tc, "linear_num_key_heads", 16);
  q.gdn_v_heads = geti(tc, "linear_num_value_heads", 32);
  q.gdn_k_dim   = geti(tc, "linear_key_head_dim", 128);
  q.gdn_v_dim   = geti(tc, "linear_value_head_dim", 128);
  q.gdn_conv_dim = 2 * q.gdn_k_heads * q.gdn_k_dim +
                   q.gdn_v_heads * q.gdn_v_dim;
  q.n_experts        = geti(tc, "num_experts", 256);
  q.top_k            = geti(tc, "num_experts_per_tok", 8);
  q.moe_inner        = geti(tc, "moe_intermediate_size", 512);
  q.moe_shared_inner = geti(tc, "shared_expert_intermediate_size", 512);
  q.moe_norm_topk    = getb(tc, "norm_topk_prob", true);
  q.tie_embeddings   = false;
  q.use_bf16         = true;
  q.dense            = false;
  q.zero_centered_norm = true;
  q.attn_output_gate = true;
  q.qk_norm          = true;
  q.backbone_only    = true;             // the embedding and heads are ours
  q.weight_prefix    = "model.language_model.";
  q.model_seg        = "";
  // The routed experts' quantization: block FP8 (fishaudio/drama-3), or
  // GPTQ int4, symmetric, group 128, no act order, for the experts and the
  // attention / gated-DeltaNet projections (fishaudio/drama-3-int4).
  c.experts = "fp8";
  if (obj.contains("quantization_config") &&
      obj.at("quantization_config").is_object()) {
    const FlexData qcfd = obj.at("quantization_config");
    auto qc = qcfd.as_object();
    const std::string m = str(qc, "quant_method");
    auto gptq4 = [&](const char* part, const char* method) {
      if (!qc.contains(part) || !qc.at(part).is_object()) { return false; }
      const FlexData pfd = qc.at(part);
      auto po = pfd.as_object();
      return str(po, "quant_method") == method && geti(po, "bits", 0) == 4 &&
             geti(po, "group_size", 0) == 128 && getb(po, "sym", false) &&
             !getb(po, "desc_act", false);
    };
    if (m == "drama3-int4") {
      if (!gptq4("experts", "gptq") ||
          (qc.contains("dense") && !gptq4("dense", "gptq_marlin"))) {
        return fail("quantization_config drama3-int4: this port reads GPTQ "
                    "int4, symmetric, group 128, no act order");
      }
      c.experts = "int4";
    } else if (m != "bsgemm-moe") {
      return fail("quant_method \"" + m + "\": this port reads "
                  "fishaudio/drama-3 (bsgemm-moe, block-FP8 experts) and "
                  "fishaudio/drama-3-int4 (drama3-int4, GPTQ)");
    }
  }

  // ---- the codebook-MTP tower -------------------------------------------
  MetalDrama3Tower::Config& t = c.tower;
  if (str(dc, "decoder_type") != "codebook_mtp_shared") {
    return fail("audio_decoder_config.decoder_type \"" +
                str(dc, "decoder_type") + "\" is not codebook_mtp_shared");
  }
  if (getb(dc, "use_moe", false) || !getb(dc, "rotary_interleaved", true) ||
      !getb(dc, "attention_output_gate", true) ||
      getb(dc, "mtp_use_swa", false) ||
      !getb(dc, "layernorm_zero_centered_gamma", true)) {
    return fail("audio_decoder_config departs from Drama 3's tower (dense "
                "MLP, interleaved rotary, output gate, full attention)");
  }
  t.dim           = geti(dc, "dim", 2048);
  t.n_head        = geti(dc, "n_head", 16);
  t.n_kv          = geti(dc, "n_local_heads", 2);
  t.head_dim      = geti(dc, "head_dim", 256);
  t.ffn           = geti(dc, "intermediate_size", 6144);
  t.eps           = getf(dc, "norm_eps", 1e-6f);
  t.num_codebooks = geti(dc, "num_codebooks", 10);
  t.codebook_size = geti(dc, "vocab_size", 4096);
  t.shared_layers = geti(dc, "shared_enrichment_layers", 4);
  t.rope_base     = getf(dc, "rope_base", 1.0e7f);
  t.rotary_dim    = (int)((float)t.head_dim *
                          getf(dc, "rotary_percent", 0.25f));
  t.backbone_codebooks = geti(dc, "vq_backbone_codebooks", 1);
  t.sem0 = c.semantic_begin;
  t.sem1 = c.semantic_end;
  if (t.backbone_codebooks != 1 || t.dim != q.hidden ||
      c.semantic_end - c.semantic_begin + 1 != t.codebook_size ||
      t.num_codebooks + 1 != MetalDrama3Model::kFrame) {
    return fail("an unexpected Drama 3 geometry (backbone codebooks, "
                "hidden, semantic band, codebook count)");
  }
  *out = c;
  return true;
}

bool
Drama3Config::is_drama3_dir(const std::string& dir)
{
  Drama3Config c;
  return from_dir(dir, &c, nullptr);
}

// ---- load -----------------------------------------------------------------

MetalDrama3Model::~MetalDrama3Model() = default;

std::unique_ptr<MetalDrama3Model>
MetalDrama3Model::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                       const Drama3Config& cfg, const LoadOptions& opt,
                       std::string* err)
{
  if (ws == nullptr || mc == nullptr) {
    if (err != nullptr) { *err = "no checkpoint or no metal device"; }
    return nullptr;
  }
  auto m = std::unique_ptr<MetalDrama3Model>(new MetalDrama3Model());
  m->_ws = std::move(ws);
  m->_mc = mc;
  m->_cfg = cfg;
  m->_opt = opt;
  if (!m->init_(err)) { return nullptr; }
  return m;
}

bool
MetalDrama3Model::init_(std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = "[drama3] " + std::move(m); }
    return false;
  };
  const Drama3Config& c = _cfg;
  MetalCompute* mc = _mc;
  MetalQwenModel::Config q = c.backbone;
  q.max_seq = _opt.max_rows + 256;
  q.expert_cache_bytes = _opt.expert_cache_bytes;
  q.autotune = _opt.autotune;
  _lm = MetalQwenModel::load(_ws, mc, q);
  if (!_lm) { return fail("the backbone did not load"); }
  std::string terr;
  _tower = MetalDrama3Tower::load(_ws, mc, c.tower, &terr);
  if (!_tower) { return fail(terr); }

  const MetalLlamaWeights& src = _ws->src();
  const int H = q.hidden;
  const std::size_t row = (std::size_t)H * 2;
  const int V = c.tower.codebook_size;
  // Both vocabulary-wide tables are read by row from their files: the
  // embedding as prompts need it, the head once for its candidates.
  if (!open_rows_(_ws->dir(), "model.language_model.embed_tokens.weight",
                  src, &_embed_rows) ||
      _embed_rows.row != row || _embed_rows.rows < c.semantic_end + 1) {
    return fail("model.language_model.embed_tokens is missing or "
                "misshapen");
  }
  _embed_sem = mc->make_shared_buffer((std::size_t)V * row);
  if (_embed_sem.empty() ||
      !_embed_rows.read(c.semantic_begin, V, _embed_sem.contents())) {
    return fail("could not read the embeddings' semantic band");
  }
  // The main head: lm_head rows [eos, semantic 1 .. end].
  {
    RowFile head;
    if (!open_rows_(_ws->dir(), "lm_head.weight", src, &head) ||
        head.row != row || head.rows < c.semantic_end + 1) {
      return fail("lm_head is missing or misshapen");
    }
    const int n = c.main_candidates();
    _head = mc->make_shared_buffer((std::size_t)n * row);
    if (_head.empty()) { return fail("out of memory"); }
    auto* d = static_cast<std::uint8_t*>(_head.contents());
    if (!head.read(c.eos, 1, d) ||
        !head.read(c.semantic_begin + 1, n - 1, d + row)) {
      return fail("could not read lm_head");
    }
  }
  _lib_d3 = mc->load_library("drama3_bf16");
  _fn_embed = _lib_d3.function("drama3_embed_rows");
  _fn_sample_main = _lib_d3.function("drama3_sample_main");
  if (!_fn_embed.valid() || !_fn_sample_main.valid()) {
    return fail("the drama3 kernels did not load");
  }
  for (SharedBuffer& f : _frame) {
    f = mc->make_shared_buffer(kFrame * sizeof(std::int32_t));
  }
  _ras = mc->make_shared_buffer(kRasWindow * sizeof(std::int32_t));
  _mlogits = mc->make_shared_buffer((std::size_t)c.main_candidates() * 2);
  _hrow = mc->make_shared_buffer(row);
  _hrow_in = mc->make_shared_buffer(row);
  if (_frame[0].empty() || _frame[1].empty() || _ras.empty() ||
      _mlogits.empty() || _hrow.empty() || _hrow_in.empty()) {
    return fail("out of memory");
  }
  return true;
}

std::size_t
MetalDrama3Model::resident_bytes() const
{
  std::size_t b = _embed_sem.byte_size() + _head.byte_size();
  if (_lm) { b += (std::size_t)_lm->resident_bytes(); }
  if (_tower) { b += _tower->resident_bytes(); }
  return b;
}

std::size_t
MetalDrama3Model::weight_bytes(const std::string& dir)
{
  auto w = MetalLlamaWeights::open_model(dir);
  if (!w.has_value()) { return 0; }
  std::size_t n = 0, row = 0;
  auto ends = [](const std::string& s, std::string_view t) {
    return s.size() > t.size() &&
           s.compare(s.size() - t.size(), t.size(), t) == 0;
  };
  for (const std::string& nm : w->tensor_names()) {
    const auto* ti = w->info(nm);
    if (ti == nullptr) { continue; }
    if (nm == "lm_head.weight" ||
        nm == "model.language_model.embed_tokens.weight") {
      if (ti->shape.size() == 2) { row = (std::size_t)ti->shape[1] * 2; }
      continue;
    }
    // GPTQ int4, as held: an expert in the affine w4 g64 layout (codes +
    // a bf16 scale and bias per 64: 9/8 of its codes), a dense projection
    // dequantized to bf16 (4x). Its scales, zeros and g_idx are consumed.
    if (ends(nm, ".qweight")) {
      const bool expert = nm.find(".mlp.experts.") != std::string::npos;
      n += expert ? (std::size_t)ti->nbytes * 9 / 8
                  : (std::size_t)ti->nbytes * 4;
      continue;
    }
    if (ends(nm, ".scales") || ends(nm, ".qzeros") || ends(nm, ".g_idx")) {
      continue;
    }
    n += (std::size_t)ti->nbytes;
  }
  // The head's candidates and the embedding's semantic band.
  return n + 2 * (std::size_t)4096 * row;
}

std::size_t
MetalDrama3Model::routed_expert_bytes(const std::string& dir,
                                      std::size_t* per_layer)
{
  if (per_layer != nullptr) { *per_layer = 0; }
  Drama3Config c;
  if (!Drama3Config::from_dir(dir, &c)) { return 0; }
  auto w = MetalLlamaWeights::open_model(dir);
  if (!w.has_value()) { return 0; }
  const std::size_t n = MetalQwenModel::routed_expert_bytes(*w, c.backbone);
  if (per_layer != nullptr && c.backbone.n_layers > 0) {
    *per_layer = n / (std::size_t)c.backbone.n_layers;
  }
  return n;
}

std::size_t
MetalDrama3Model::run_bytes(const Drama3Config& c, int rows, int prompt_rows)
{
  const MetalQwenModel::Config& q = c.backbone;
  const std::size_t R = (std::size_t)std::max(rows, 1);
  const std::size_t P = (std::size_t)std::max(prompt_rows, 1);
  int full = 0;
  for (int L = 0; L < q.n_layers; ++L) { full += q.layer_is_full(L) ? 1 : 0; }
  // Backbone K/V (bf16, pages of 256) + the gated-DeltaNet layers' f32
  // state and conv tail.
  const std::size_t pages = (R + 255) / 256 * 256;
  const std::size_t kv = pages * (std::size_t)full * q.n_kv_heads *
                         q.head_dim * 2 * 2;
  const std::size_t gdn = (std::size_t)(q.n_layers - full) *
                          ((std::size_t)q.gdn_v_heads * q.gdn_k_dim *
                               q.gdn_v_dim * 4 +
                           (std::size_t)q.gdn_conv_dim * q.gdn_conv_kernel *
                               2);
  const std::size_t tower =
      MetalDrama3Tower::kv_bytes_for(c.tower, (int)((R + 255) / 256 * 256));
  // The prefill's widest transients: the MoE pairs' activations and
  // partials, the attention / GDN scratch and the hidden rows the tower
  // reads, all bf16 over the prompt -- and the tower's own row scratch.
  const std::size_t H = (std::size_t)q.hidden;
  const std::size_t np = P * (std::size_t)q.top_k;
  const std::size_t moe = np * ((std::size_t)q.moe_inner + H) * 2 +
                          P * (std::size_t)q.n_experts * 2;
  const std::size_t act = P * (std::size_t)(2 * q.qd() + 2 * q.kd() +
                                            q.gdn_conv_dim +
                                            q.value_dim() + 8 * H) * 2;
  const std::size_t trows = P * ((std::size_t)c.tower.qkv_width() +
                                 2 * (std::size_t)c.tower.ffn + 16 * H) * 2;
  return kv + gdn + tower + moe + act + trows;
}

MetalDrama3Model::RowFile::~RowFile()
{
  if (fd >= 0) { ::close(fd); }
}

bool
MetalDrama3Model::RowFile::read(std::int64_t r0, std::int64_t n,
                                void* dst) const
{
  if (fd < 0 || r0 < 0 || n < 0 || r0 + n > rows) { return false; }
  std::size_t left = (std::size_t)n * row;
  auto* p = static_cast<std::uint8_t*>(dst);
  off_t at = (off_t)(base + (std::uint64_t)r0 * row);
  while (left > 0) {
    const ssize_t got = ::pread(fd, p, left, at);
    if (got <= 0) { return false; }
    p += got;
    at += got;
    left -= (std::size_t)got;
  }
  return true;
}

// A [rows, cols] bf16 tensor of a safetensors checkpoint, opened for reads
// by row: the shard from the index (or the one model.safetensors), the
// tensor's offset after that shard's header.
bool
MetalDrama3Model::open_rows_(const std::string& dir, const std::string& name,
                             const MetalLlamaWeights& src, RowFile* out)
{
  namespace fs = std::filesystem;
  const auto* ti = src.info(name);
  if (ti == nullptr || ti->dtype != "BF16" || ti->shape.size() != 2) {
    return false;
  }
  std::string file = "model.safetensors";
  {
    std::ifstream in(fs::path(dir) / "model.safetensors.index.json");
    if (in) {
      try {
        const FlexData idx = FlexData::from_json(in);
        const FlexData wm = idx.as_object().at("weight_map");
        auto o = wm.as_object();
        if (o.contains(name)) { file = std::string(o.at(name).as_string()); }
      } catch (...) {
        return false;
      }
    }
  }
  const std::string path = (fs::path(dir) / file).string();
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) { return false; }
  std::uint64_t hlen = 0;
  if (::pread(fd, &hlen, 8, 0) != 8) {
    ::close(fd);
    return false;
  }
  if (out->fd >= 0) { ::close(out->fd); }
  out->fd = fd;
  out->base = 8 + hlen + ti->offset;
  out->rows = ti->shape[0];
  out->row = (std::size_t)ti->shape[1] * 2;
  if (ti->nbytes != (std::uint64_t)out->rows * out->row) {
    ::close(fd);
    out->fd = -1;
    return false;
  }
  return true;
}

SharedBuffer
MetalDrama3Model::embed_(const std::int32_t* frames, int n)
{
  const int H = _cfg.backbone.hidden;
  const std::size_t row = (std::size_t)H * 2;
  SharedBuffer rows = _mc->make_shared_buffer((std::size_t)n * row);
  SharedBuffer fr = _mc->make_shared_buffer((std::size_t)n * kFrame * 4);
  SharedBuffer x = _mc->make_shared_buffer((std::size_t)n * row);
  if (rows.empty() || fr.empty() || x.empty()) { return {}; }
  auto* dst = static_cast<std::uint8_t*>(rows.contents());
  for (int r = 0; r < n; ++r) {
    const int tok = frames[(std::size_t)r * kFrame];
    if (tok < 0 || tok >= _cfg.text_vocab ||
        !_embed_rows.read(tok, 1, dst + (std::size_t)r * row)) {
      return {};
    }
  }
  std::memcpy(fr.contents(), frames, (std::size_t)n * kFrame * 4);
  CommandStream s = _mc->make_command_stream();
  {
    ComputeEncoder enc = s.begin_compute();
    enc.set_function(_fn_embed);
    enc.set_buffer(0, fr);
    enc.set_buffer(1, rows);
    enc.set_buffer(2, _tower->codebook_embeddings());
    enc.set_buffer(3, x);
    enc.set_constant(4, H);
    enc.set_constant(5, _cfg.semantic_begin);
    enc.set_constant(6, _cfg.semantic_end);
    enc.set_constant(7, _cfg.tower.codebook_size);
    enc.set_constant(8, 1);
    enc.set_constant(9, 0);
    enc.set_constant(10, n);
    enc.dispatch({(unsigned)H, (unsigned)n, 1}, {256, 1, 1});
  }
  if (!s.commit().wait_ok()) { return {}; }
  return x;
}

namespace {

// Read `n` compute-dtype (bf16) values into f32.
void read_bf16(const SharedBuffer& b, std::size_t off_elems, std::size_t n,
               float* out)
{
  const auto* s = static_cast<const std::uint16_t*>(b.contents()) + off_elems;
  for (std::size_t i = 0; i < n; ++i) { out[i] = bf16_f32(s[i]); }
}

}  // namespace

bool
MetalDrama3Model::teacher_forced(const std::vector<std::int32_t>& frames,
                                 int n, std::vector<float>* hidden,
                                 std::vector<float>* main_logits,
                                 std::vector<float>* head_logits,
                                 std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = "[drama3] " + std::move(m); }
    return false;
  };
  const Drama3Config& c = _cfg;
  if (n <= 0 || frames.size() < (std::size_t)n * kFrame) {
    return fail("teacher_forced: too few frames");
  }
  const int H = c.backbone.hidden, V = c.tower.codebook_size;
  const int NC = c.main_candidates();
  SharedBuffer x = embed_(frames.data(), n);
  if (x.empty()) { return fail("embedding failed"); }
  ContextManager* cm = _lm->context_manager();
  const ContextId cid = cm->acquire_root();
  SharedBuffer hid = _lm->forward_embeddings_hidden_all(cid, x, n);
  cm->release(cid);
  if (hid.empty()) { return fail("backbone forward failed"); }
  if (!_tower->reserve(n)) { return fail("tower K/V allocation failed"); }
  SharedBuffer fr = _mc->make_shared_buffer((std::size_t)n * kFrame * 4);
  SharedBuffer lab = _mc->make_shared_buffer((std::size_t)n * kFrame * 4);
  SharedBuffer ml = _mc->make_shared_buffer((std::size_t)n * NC * 2);
  SharedBuffer hl = _mc->make_shared_buffer(
      (std::size_t)(c.tower.num_codebooks - 1) * n * V * 2);
  if (fr.empty() || lab.empty() || ml.empty() || hl.empty()) {
    return fail("out of memory");
  }
  std::memcpy(fr.contents(), frames.data(), (std::size_t)n * kFrame * 4);
  auto* lp = static_cast<std::int32_t*>(lab.contents());
  std::memcpy(lp, frames.data() + kFrame, (std::size_t)(n - 1) * kFrame * 4);
  std::fill(lp + (std::size_t)(n - 1) * kFrame, lp + (std::size_t)n * kFrame,
            0);
  lp[(std::size_t)(n - 1) * kFrame] = c.eos;
  CommandStream s = _mc->make_command_stream();
  {
    ComputeEncoder enc = s.begin_compute();
    _tower->encode_gemm(enc, _head, hid, 0, ml, 0, H, NC, n);
    if (!_tower->encode_rows(enc, hid, fr, lab, n, 0, &hl)) {
      return fail("tower encode failed");
    }
  }
  std::string werr;
  if (!s.commit().wait_ok(&werr)) { return fail(werr); }
  if (hidden != nullptr) {
    hidden->resize((std::size_t)n * H);
    read_bf16(hid, 0, hidden->size(), hidden->data());
  }
  if (main_logits != nullptr) {
    main_logits->resize((std::size_t)n * NC);
    read_bf16(ml, 0, main_logits->size(), main_logits->data());
  }
  if (head_logits != nullptr) {
    head_logits->resize((std::size_t)(c.tower.num_codebooks - 1) * n * V);
    read_bf16(hl, 0, head_logits->size(), head_logits->data());
  }
  return true;
}

bool
MetalDrama3Model::teacher_decode(const std::vector<std::int32_t>& frames,
                                 int n, int n_prompt,
                                 std::vector<float>* main_logits,
                                 std::vector<float>* head_logits,
                                 std::string* err,
                                 std::vector<float>* hidden)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = "[drama3] " + std::move(m); }
    return false;
  };
  const Drama3Config& c = _cfg;
  const int P = n_prompt;
  if (P < 2 || n <= P || frames.size() < (std::size_t)n * kFrame) {
    return fail("teacher_decode: needs a prompt of 2+ rows and a frame");
  }
  const int H = c.backbone.hidden, V = c.tower.codebook_size;
  const int NC = c.main_candidates(), NH = c.tower.num_codebooks - 1;
  const int rows = n - P + 1;          // logits rows P-1 .. n-1
  SharedBuffer x = embed_(frames.data(), P);
  if (x.empty()) { return fail("embedding failed"); }
  ContextManager* cm = _lm->context_manager();
  const ContextId cid = cm->acquire_root();
  struct Rel {
    ContextManager* cm; ContextId c;
    ~Rel() { cm->release(c); }
  } rel{cm, cid};
  SharedBuffer hid = _lm->forward_embeddings_hidden_all(cid, x, P);
  if (hid.empty()) { return fail("backbone prefill failed"); }
  if (!_tower->reserve(n)) { return fail("tower K/V allocation failed"); }
  SharedBuffer fr = _mc->make_shared_buffer((std::size_t)n * kFrame * 4);
  SharedBuffer lab = _mc->make_shared_buffer((std::size_t)n * kFrame * 4);
  SharedBuffer hl = _mc->make_shared_buffer((std::size_t)NH * V * 2);
  if (fr.empty() || lab.empty() || hl.empty()) { return fail("out of memory"); }
  std::memcpy(fr.contents(), frames.data(), (std::size_t)n * kFrame * 4);
  auto* lp = static_cast<std::int32_t*>(lab.contents());
  std::memcpy(lp, frames.data() + kFrame, (std::size_t)(n - 1) * kFrame * 4);
  std::fill(lp + (std::size_t)(n - 1) * kFrame, lp + (std::size_t)n * kFrame,
            0);
  lp[(std::size_t)(n - 1) * kFrame] = c.eos;
  // The prompt's rows but its last, teacher-forced in one pass.
  {
    CommandStream s = _mc->make_command_stream();
    {
      ComputeEncoder enc = s.begin_compute();
      if (!_tower->encode_rows(enc, hid, fr, lab, P - 1, 0)) {
        return fail("tower prefill failed");
      }
    }
    if (!s.commit().wait_ok()) { return fail("tower prefill failed"); }
  }
  std::memcpy(_hrow.contents(),
              static_cast<const std::uint8_t*>(hid.contents()) +
                  (std::size_t)(P - 1) * H * 2,
              (std::size_t)H * 2);
  if (main_logits != nullptr) {
    main_logits->assign((std::size_t)rows * NC, 0);
  }
  if (head_logits != nullptr) {
    head_logits->assign((std::size_t)NH * rows * V, 0);
  }
  if (hidden != nullptr) { hidden->assign((std::size_t)rows * H, 0); }
  // Row r = P-1 .. n-1: its hidden (prefill for P-1, else a decode step on
  // row r's embedding), then the main head and a one-row tower.
  CommandStream s = _mc->make_command_stream();
  const SharedBuffer* last_hb = nullptr;
  for (int r = P - 1; r < n; ++r) {
    // Row r's input frame and its label, as the one-row tower reads them.
    std::memcpy(_frame[0].contents(),
                frames.data() + (std::size_t)r * kFrame, kFrame * 4);
    std::memcpy(_frame[1].contents(), lp + (std::size_t)r * kFrame,
                kFrame * 4);
    {
      ComputeEncoder enc = s.begin_compute();
      const SharedBuffer* hb = &_hrow;
      if (r > P - 1) {
        // Row r's backbone input, written where the decode step reads it:
        // an audio row from the resident semantic band, as generation
        // embeds; any other (the closing end-of-speech row) from the
        // mapped table.
        SharedBuffer& din = _lm->decode_input_buffer();
        const int tok = frames[(std::size_t)r * kFrame];
        const bool sem = tok >= c.semantic_begin && tok <= c.semantic_end;
        if (!sem && !_embed_rows.read(tok, 1, _hrow_in.contents())) {
          return fail("could not read an embedding row");
        }
        enc.set_function(_fn_embed);
        enc.set_buffer(0, _frame[0]);
        enc.set_buffer(1, sem ? _embed_sem : _hrow_in);
        enc.set_buffer(2, _tower->codebook_embeddings());
        enc.set_buffer(3, din);
        enc.set_constant(4, H);
        enc.set_constant(5, c.semantic_begin);
        enc.set_constant(6, c.semantic_end);
        enc.set_constant(7, V);
        enc.set_constant(8, sem ? 0 : 1);
        enc.set_constant(9, c.semantic_begin);
        enc.set_constant(10, V);
        enc.dispatch({(unsigned)H, 1, 1}, {256, 1, 1});
        hb = _lm->encode_decode_prewritten(enc, cid);
        if (hb == nullptr) { return fail("decode step failed"); }
        last_hb = hb;
      }
      _tower->encode_gemm(enc, _head, *hb, 0, _mlogits, 0, H, NC, 1);
      if (!_tower->encode_rows(enc, *hb, _frame[0], _frame[1], 1, r, &hl)) {
        return fail("tower step failed");
      }
    }
    std::string werr;
    if (!s.commit().wait_ok(&werr)) { return fail(werr); }
    const int k = r - (P - 1);
    if (hidden != nullptr) {
      read_bf16(r > P - 1 ? *last_hb : _hrow, 0, (std::size_t)H,
                hidden->data() + (std::size_t)k * H);
    }
    if (main_logits != nullptr) {
      read_bf16(_mlogits, 0, (std::size_t)NC,
                main_logits->data() + (std::size_t)k * NC);
    }
    if (head_logits != nullptr) {
      for (int j = 0; j < NH; ++j) {
        read_bf16(hl, (std::size_t)j * V, (std::size_t)V,
                  head_logits->data() + ((std::size_t)j * rows + k) * V);
      }
    }
  }
  return true;
}

bool
MetalDrama3Model::generate(const std::vector<std::int32_t>& prompt,
                           int n_prompt, const Sampling& sp,
                           const std::function<bool(const Frame&, int)>&
                               on_frame,
                           std::vector<Frame>* frames, std::string* err,
                           std::vector<float>* main_logits)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = "[drama3] " + std::move(m); }
    return false;
  };
  const Drama3Config& c = _cfg;
  const int P = n_prompt;
  if (P < 2 || prompt.size() < (std::size_t)P * kFrame) {
    return fail("generate: the prompt needs at least two rows");
  }
  if (main_logits != nullptr) { main_logits->clear(); }
  const int H = c.backbone.hidden, V = c.tower.codebook_size;
  const int NC = c.main_candidates();
  const int max_frames = std::max(sp.max_frames, 1);
  if (P + max_frames > _opt.max_rows) {
    return fail(std::format("a {}-row prompt and {} frames exceed the {} "
                            "rows this model was loaded for", P, max_frames,
                            _opt.max_rows));
  }
  using clock = std::chrono::steady_clock;
  auto msec = [](clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };
  const auto t0 = clock::now();
  _stats = RunStats{};
  _stats.prompt_rows = P;
  SharedBuffer x = embed_(prompt.data(), P);
  if (x.empty()) { return fail("embedding failed"); }
  _stats.embed_ms = msec(clock::now() - t0);
  ContextManager* cm = _lm->context_manager();
  const ContextId cid = cm->acquire_root();
  struct Rel {
    ContextManager* cm; ContextId c;
    ~Rel() { cm->release(c); }
  } rel{cm, cid};
  const auto t_bb = clock::now();
  SharedBuffer hid = _lm->forward_embeddings_hidden_all(cid, x, P);
  if (hid.empty()) { return fail("backbone prefill failed"); }
  _stats.backbone_ms = msec(clock::now() - t_bb);
  if (!_tower->reserve(P + max_frames)) {
    return fail("tower K/V allocation failed");
  }
  // The prompt's rows but its last through the tower, teacher-forced.
  {
    SharedBuffer fr = _mc->make_shared_buffer((std::size_t)P * kFrame * 4);
    SharedBuffer lab = _mc->make_shared_buffer((std::size_t)P * kFrame * 4);
    if (fr.empty() || lab.empty()) { return fail("out of memory"); }
    std::memcpy(fr.contents(), prompt.data(), (std::size_t)P * kFrame * 4);
    std::memcpy(lab.contents(), prompt.data() + kFrame,
                (std::size_t)(P - 1) * kFrame * 4);
    CommandStream s = _mc->make_command_stream();
    {
      ComputeEncoder enc = s.begin_compute();
      if (!_tower->encode_rows(enc, hid, fr, lab, P - 1, 0)) {
        return fail("tower prefill failed");
      }
    }
    if (!s.commit().wait_ok()) { return fail("tower prefill failed"); }
  }
  std::memcpy(_hrow.contents(),
              static_cast<const std::uint8_t*>(hid.contents()) +
                  (std::size_t)(P - 1) * H * 2,
              (std::size_t)H * 2);
  // Frame state: _frame[cur] is the input row of this step, the other the
  // frame being drawn. The repetition window starts as the prompt's last
  // kRasWindow main tokens, oldest first, padded in front with -1 -- as
  // every reference seeds it: a short text after a cloned voice (or an
  // earlier take) has that voice's last codes inside the window.
  int cur = 0;
  std::memcpy(_frame[cur].contents(),
              prompt.data() + (std::size_t)(P - 1) * kFrame, kFrame * 4);
  {
    auto* ras = static_cast<std::int32_t*>(_ras.contents());
    const int keep = std::min(P, kRasWindow);
    std::fill_n(ras, kRasWindow - keep, -1);
    for (int i = 0; i < keep; ++i) {
      ras[kRasWindow - keep + i] =
          prompt[(std::size_t)(P - keep + i) * kFrame];
    }
  }
  MetalDrama3Tower::Draw dr;
  dr.temperature = sp.temperature;
  dr.top_k = sp.top_k;
  dr.top_p = sp.top_p;
  dr.seed = sp.seed;
  if (frames != nullptr) { frames->clear(); }
  const auto t_pre = clock::now();
  _stats.prefill_ms = msec(t_pre - t0);
  auto t_first = t_pre;
  CommandStream s = _mc->make_command_stream();
  int drawn = 0;
  for (int i = 0; i < max_frames; ++i) {
    const int pos = P - 1 + i;
    SharedBuffer& in = _frame[cur];
    SharedBuffer& out = _frame[1 - cur];
    {
      ComputeEncoder enc = s.begin_compute();
      const SharedBuffer* hb = &_hrow;
      if (i > 0) {
        hb = _lm->encode_decode_prewritten(enc, cid);
        if (hb == nullptr) { return fail("decode step failed"); }
      }
      _tower->encode_gemm(enc, _head, *hb, 0, _mlogits, 0, H, NC, 1);
      enc.set_function(_fn_sample_main);
      enc.set_buffer(0, _mlogits);
      enc.set_buffer(1, out);
      enc.set_buffer(2, _ras);
      enc.set_constant(3, NC);
      enc.set_constant(4, sp.temperature);
      enc.set_constant(5, std::clamp(sp.top_k, 1, kMaxRank));
      enc.set_constant(6, sp.top_p);
      enc.set_constant(7, sp.seed);
      enc.set_constant(8, (std::uint32_t)i);
      enc.set_constant(9, c.eos);
      enc.set_constant(10, c.semantic_begin);
      enc.set_constant(11, kRasWindow);
      enc.set_constant(12, kMaxRank);
      enc.dispatch({256, 1, 1}, {256, 1, 1});
      dr.frame = (std::uint32_t)i;
      if (!_tower->encode_step(enc, *hb, in, out, pos, dr)) {
        return fail("tower step failed");
      }
      // The next step's backbone input, from the frame just drawn.
      enc.set_function(_fn_embed);
      enc.set_buffer(0, out);
      enc.set_buffer(1, _embed_sem);
      enc.set_buffer(2, _tower->codebook_embeddings());
      enc.set_buffer(3, _lm->decode_input_buffer());
      enc.set_constant(4, H);
      enc.set_constant(5, c.semantic_begin);
      enc.set_constant(6, c.semantic_end);
      enc.set_constant(7, V);
      enc.set_constant(8, 0);
      enc.set_constant(9, c.semantic_begin);
      enc.set_constant(10, V);
      enc.dispatch({(unsigned)H, 1, 1}, {256, 1, 1});
    }
    std::string werr;
    if (!s.commit().wait_ok(&werr)) { return fail(werr); }
    if (i == 0) {
      t_first = clock::now();
      _stats.first_ms = msec(t_first - t_pre);
    }
    if (main_logits != nullptr) {
      const std::size_t o = main_logits->size();
      main_logits->resize(o + (std::size_t)NC);
      read_bf16(_mlogits, 0, (std::size_t)NC, main_logits->data() + o);
    }
    const auto* f = static_cast<const std::int32_t*>(out.contents());
    if (f[0] == c.eos) { _stats.ended = true; break; }
    Frame fc;
    std::copy(f + 1, f + kFrame, fc.begin());
    if (frames != nullptr) { frames->push_back(fc); }
    ++drawn;
    if (on_frame && !on_frame(fc, i)) { break; }
    cur = 1 - cur;
  }
  _stats.frames = drawn;
  const double rest = msec(clock::now() - t_first);
  _stats.frames_per_s = drawn > 1 && rest > 0.0 ? (drawn - 1) * 1e3 / rest
                                                 : 0.0;
  return true;
}

}  // namespace vpipe::genai
